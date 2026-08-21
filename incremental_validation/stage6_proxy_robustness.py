from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import argparse, csv, json, platform, sys
import numpy as np
import torch

from author_baseline.recognizer import OneHotArchive
from author_baseline.weights import DEFAULT_WEIGHT_ROOT, EXPECTED_SHA256, build_primary_type_recognizer
from hierarchical_ecc.coding import stable_seed
from hierarchical_ecc.config import ExperimentConfig, KNOWN_CODE_TYPES, NO_ECC_TYPES
from hierarchical_ecc.data import ReferenceFactory
from incremental_validation.collector import TorchPresenceDetector
from incremental_validation.comparison import IncrementalThresholds, KNOWN_TYPES
from incremental_validation.embedding_rejection import extract_archive_embeddings
from incremental_validation.inner_codes import archives_from_references, generate_inner_code_references
from incremental_validation.simulation import ExternalPresenceCNN, audit_molecular_references
from incremental_validation.stage2_feature_rejection import _auroc, _average_precision, _fpr_at_95_tpr, _macro_f1, _sha256, acceptance_threshold
from incremental_validation.stage3_multidetector import PROXY_FAMILIES, generate_proxy_references
from incremental_validation.stage4_conservative_robustness import ERROR_RATES, Q_VALUES, M_VALUES, SEEDS
from incremental_validation.stage5_structural_proxy import ProxyClassifier, archive_feature_blocks, three_state_consensus

CATEGORIES=(*KNOWN_CODE_TYPES,*NO_ECC_TYPES,"HEDGES","DNA-Aeon")
BLOCKS={"sequence":(0,165),"embedding":(165,421),"logits":(421,464)}
ABLATIONS={"A_sequence":("sequence",),"B_embedding":("embedding",),"C_logits":("logits",),"D_sequence_embedding":("sequence","embedding"),"E_sequence_logits":("sequence","logits"),"F_embedding_logits":("embedding","logits"),"G_all":("sequence","embedding","logits")}
SIX=("no_ecc","unknown_ecc",*KNOWN_TYPES); SEVEN=("no_ecc","uncertain_ecc","unknown_ecc",*KNOWN_TYPES)


def select_blocks(x:np.ndarray,names:Sequence[str])->np.ndarray:
    return np.concatenate([np.asarray(x)[...,BLOCKS[n][0]:BLOCKS[n][1]] for n in names],axis=-1)


def prefix_archive(a:OneHotArchive,M:int,q:int)->OneHotArchive:
    return OneHotArchive(np.asarray(a.one_hot)[:M,:q],np.asarray(a.mask)[:M,:q])


def make_refs(cfg:ExperimentConfig,seed:int,out:Path)->tuple[dict[str,np.ndarray],dict[str,Any]]:
    factory=ReferenceFactory(cfg); sets={}; namespace=f"stage6-test-seed-{seed}"
    for cat in (*KNOWN_CODE_TYPES,*NO_ECC_TYPES):
        sets[cat]=np.stack([factory.make_reference(cat,namespace,i,0) for i in range(2500)])
    inner=[]
    for cat in ("HEDGES","DNA-Aeon"):
        refs,val=generate_inner_code_references(cat,2500,seed,out/f"{cat.lower().replace('-','_')}.fasta",namespace=namespace)
        sets[cat]=refs; inner.append(val.__dict__)
    return sets,{"namespace":namespace,"audit":audit_molecular_references(sets),"inner":inner}


def make_known_refs(cfg:ExperimentConfig,seed:int,split:str,count:int)->dict[str,np.ndarray]:
    f=ReferenceFactory(cfg); ns=f"stage6-{split}-seed-{seed}"
    return {c:np.stack([f.make_reference(c,ns,i,0) for i in range(count*20)]) for c in KNOWN_CODE_TYPES}


def flatten_archives(cfg,sets,split,archives,M,q,error):
    result=[]; cats=[]; ids=[]
    for c in sets:
        result.extend(archives_from_references(cfg,c,split,sets[c][:archives*M],archives,M,q,error)); cats += [c]*archives; ids += [f"{split}:{c}:{i}" for i in range(archives)]
    return result,np.asarray(cats),np.asarray(ids)


def score_archives(task,presence,classifier,archives,cats,ids,prefixes=())->dict[str,np.ndarray]:
    feats=[]; proxy=[]; energy=[]; closed=[]; ecc=[]
    ps={name:[] for name,_,_ in prefixes}; pe={name:[] for name,_,_ in prefixes}; pc={name:[] for name,_,_ in prefixes}; pen={name:[] for name,_,_ in prefixes}; pf={name:[] for name,_,_ in prefixes if name=='M20'}
    for i,a in enumerate(archives):
        logits,emb=extract_archive_embeddings(task,a); b=archive_feature_blocks(a,logits,emb); x=np.concatenate((b["sequence"],b["embedding"],b["logits"])); feats.append(x); proxy.append(classifier.score(x[None])[0])
        mx=logits.max(-1,keepdims=True); energy.append(float((-(mx[...,0]+np.log(np.exp(logits-mx).sum(-1)))).mean())); p=np.exp(logits-mx); p/=p.sum(-1,keepdims=True); closed.append(int(p.mean(1).mean(0).argmax())); pres=presence.predict_probabilities(a); ecc.append(float(pres.mean()))
        for name,M,q in prefixes:
            aa=prefix_archive(a,M,q); ll=logits[:M,:q]; ee=emb[:M,:q]; bb=archive_feature_blocks(aa,ll,ee); xx=np.concatenate((bb["sequence"],bb["embedding"],bb["logits"])); ps[name].append(classifier.score(xx[None])[0]); mm=ll.max(-1,keepdims=True); pp=np.exp(ll-mm); pp/=pp.sum(-1,keepdims=True); pc[name].append(int(pp.mean(1).mean(0).argmax())); pe[name].append(float(pres[:M,:q].mean())); pen[name].append(float((-(mm[...,0]+np.log(np.exp(ll-mm).sum(-1)))).mean()));
            if name=='M20': pf[name].append(xx)
        if (i+1)%10==0 or i+1==len(archives): print(f"stage6 score {i+1}/{len(archives)}",flush=True)
    out={"features":np.stack(feats),"proxy":np.asarray(proxy),"energy":np.asarray(energy),"closed":np.asarray(closed),"ecc":np.asarray(ecc),"categories":cats,"archive_ids":ids}
    for name in ps: out[f"prefix_{name}"]=np.asarray(ps[name]); out[f"prefix_ecc_{name}"]=np.asarray(pe[name]); out[f"prefix_closed_{name}"]=np.asarray(pc[name]); out[f"prefix_energy_{name}"]=np.asarray(pen[name])
    if pf.get('M20'): out['prefix_features_M20']=np.stack(pf['M20'])
    return out


def outputs(data,tau1,tau_proxy):
    cats=data["categories"].astype(str); closed=np.asarray(KNOWN_TYPES,dtype=object)[data["closed"]]; out=closed.copy(); out[data["proxy"]>tau_proxy]="unknown_ecc"; out[data["ecc"]<tau1]="no_ecc"; return out.astype(str)


def metrics(data,out):
    cats=data["categories"].astype(str); known=np.isin(cats,KNOWN_TYPES); no=np.isin(cats,NO_ECC_TYPES); unk=np.isin(cats,("HEDGES","DNA-Aeon")); closed=np.asarray(KNOWN_TYPES)[data["closed"]]
    cf=_macro_f1(cats[known],closed[known]); ff=_macro_f1(cats[known],out[known]); score=data["proxy"]
    truth=[c if c in KNOWN_TYPES else ("no_ecc" if c in NO_ECC_TYPES else "unknown_ecc") for c in cats]; idx={x:i for i,x in enumerate(SIX)}; mat=np.zeros((6,6),int)
    for a,b in zip(truth,out): mat[idx[str(a)],idx[str(b)]]+=1
    return {"known_acceptance_rate":float(np.mean(np.isin(out[known],KNOWN_TYPES))),"known_no_ecc_rate":float(np.mean(out[known]=="no_ecc")),"known_proxy_rejection_rate":float(np.mean(out[known]=="unknown_ecc")),"known_type_acceptance":{c:float(np.mean(np.isin(out[cats==c],KNOWN_TYPES))) for c in KNOWN_TYPES},"closed_macro_f1":cf,"known_type_macro_f1":ff,"known_type_macro_f1_change_from_closed":ff-cf,"no_ecc_specificity":float(np.mean(out[no]=="no_ecc")),"HEDGES_unknown_recall":float(np.mean(out[cats=="HEDGES"]=="unknown_ecc")),"DNA_Aeon_unknown_recall":float(np.mean(out[cats=="DNA-Aeon"]=="unknown_ecc")),"combined_unknown_recall":float(np.mean(out[unk]=="unknown_ecc")),"unknown_direct_known_rate":float(np.mean(np.isin(out[unk],KNOWN_TYPES))),"unknown_misclassified_as_BCH_rate":float(np.mean(out[unk]=="BCH")),"AUROC":_auroc(score[unk],score[known]),"AUPR":_average_precision(np.r_[np.zeros(known.sum()),np.ones(unk.sum())],np.r_[score[known],score[unk]]),"FPR_at_95_TPR":_fpr_at_95_tpr(score[unk],score[known]),"labels":list(SIX),"six_class_confusion_matrix":mat.tolist()}


def bootstrap(cats,data,out,seed,reps=1000):
    groups={c:np.flatnonzero(cats==c) for c in CATEGORIES}; vals=[]
    for r in range(reps):
        rng=np.random.default_rng(stable_seed("stage6-bootstrap",seed,r)); ix=np.concatenate([rng.choice(v,len(v),replace=True) for v in groups.values()]); sub={k:(v[ix] if isinstance(v,np.ndarray) and v.shape and v.shape[0]==len(cats) else v) for k,v in data.items()}; m=metrics(sub,out[ix]); vals.append([m["known_acceptance_rate"],m["combined_unknown_recall"],m["known_type_macro_f1_change_from_closed"],m["no_ecc_specificity"]])
    a=np.asarray(vals); names=("known_acceptance_rate","combined_unknown_recall","known_type_macro_f1_change_from_closed","no_ecc_specificity"); return {n:{"mean":float(a[:,i].mean()),"lower95":float(np.quantile(a[:,i],.025)),"upper95":float(np.quantile(a[:,i],.975))} for i,n in enumerate(names)}


def fit_ablation(train_known,cal_known,val_known,proxy_fit,proxy_val,names):
    candidates=[]
    for dim in (16,32,48):
      for ridge in (.01,.1,1.,10.):
       folds=[]
       for held in PROXY_FAMILIES:
        dev=[f for f in PROXY_FAMILIES if f!=held]; x=np.concatenate((select_blocks(train_known,names),*[select_blocks(proxy_fit[f],names) for f in dev])); y=np.concatenate((np.zeros(len(train_known)),*[np.ones(len(proxy_fit[f])) for f in dev])); model=ProxyClassifier.fit(x,y,dim,ridge); t=acceptance_threshold(model.score(select_blocks(cal_known,names)),.98); folds.append((float(np.mean(model.score(select_blocks(val_known,names))<=t)),float(np.mean(model.score(select_blocks(proxy_val[held],names))>t))))
       candidates.append((min(x[0] for x in folds),min(x[1] for x in folds),np.mean([x[1] for x in folds]),dim,ridge,folds))
    eligible=[x for x in candidates if x[0]>=.93] or candidates; best=max(eligible,key=lambda x:(x[1],x[2],x[0],-x[3],-x[4])); x=np.concatenate((select_blocks(train_known,names),*[select_blocks(proxy_fit[f],names) for f in PROXY_FAMILIES])); y=np.concatenate((np.zeros(len(train_known)),*[np.ones(len(proxy_fit[f])) for f in PROXY_FAMILIES])); model=ProxyClassifier.fit(x,y,best[3],best[4]); t=acceptance_threshold(model.score(select_blocks(cal_known,names)),.98); return model,t,{"pca":best[3],"ridge":best[4],"lofo_min_known":best[0],"lofo_min_proxy":best[1],"lofo_mean_proxy":float(best[2]),"folds":best[5]}


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as h: w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def svg(path,title,rows,xkey,ykeys):
    w,h=900,460; colors=['#2563eb','#dc2626','#16a34a','#9333ea']; lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><text x="450" y="25" text-anchor="middle" font-family="sans-serif">{title}</text>']
    xs=sorted(set(float(r[xkey]) for r in rows));
    for j,k in enumerate(ykeys):
      pts=[]
      for i,x in enumerate(xs):
       v=np.mean([float(r[k]) for r in rows if float(r[xkey])==x]); pts.append(f'{60+i*780/max(len(xs)-1,1)},{410-v*350}')
      lines.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[j%4]}" stroke-width="2"/><text x="{80+j*180}" y="445" fill="{colors[j%4]}">{k}</text>')
    lines.append('</svg>'); path.write_text('\n'.join(lines),encoding='utf-8')


def run(source,stage5,output,device='cuda',resume=False,command=()):
    source=Path(source).resolve(); stage5=Path(stage5).resolve(); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True); cache=output/'condition_cache'; cache.mkdir(exist_ok=True); cfg=ExperimentConfig(); dev=torch.device(device if torch.cuda.is_available() else 'cpu'); models=Path('vendor/zhouph0313_DNA/models.py').resolve(); weight=Path(DEFAULT_WEIGHT_ROOT).resolve()/'type'/'transformer_model_f10.6033.pt'; mh0,wh0=_sha256(models),_sha256(weight)
    author=build_primary_type_recognizer(device=dev,batch_size=64); [p.requires_grad_(False) for p in author.code_type.model.parameters()]; pc=torch.load(source/'models'/'external_presence_cnn.pt',map_location='cpu',weights_only=True); pm=ExternalPresenceCNN(); pm.load_state_dict(pc['state_dict']); presence=TorchPresenceDetector(pm,device=dev,batch_size=64); fixed=ProxyClassifier.load(stage5/'structural_embedding_proxy_detector.npz'); tau1=IncrementalThresholds.load(source/'thresholds.json').ecc_presence; tau=.8268975481068935
    (output/'proxy_only_protocol.json').write_text(json.dumps({"main":"Stage5 frozen proxy-only","threshold":tau,"PCA":16,"L2":1.0,"seeds":list(SEEDS),"errors":list(ERROR_RATES),"q":list(Q_VALUES),"M":list(M_VALUES)},indent=2),encoding='utf-8')
    refs_by_seed={}; audits={}; recal={}; conditions={}
    for seed in SEEDS:
      sets,audit=make_refs(cfg,seed,output/'references'/f'seed{seed}'); refs_by_seed[seed]=sets; audits[str(seed)]=audit
      calsets=make_known_refs(cfg,seed,'calibration',15); ca,cc,ci=flatten_archives(cfg,calsets,f'stage6-cal-seed-{seed}',15,20,50,.05); cp=cache/f'cal_seed{seed}.npz'; cd=dict(np.load(cp,allow_pickle=False)) if resume and cp.is_file() else score_archives(author.code_type,presence,fixed,ca,cc,ci); _save=lambda p,d:np.savez_compressed(p,**d)
      if not cp.is_file():_save(cp,cd)
      recal[seed]=float(acceptance_threshold(cd['proxy'],.98))
      for er in ERROR_RATES:
       path=cache/f'seed{seed}_error{er:.2f}.npz'
       if resume and path.is_file(): data=dict(np.load(path,allow_pickle=False))
       else:
        M=50 if np.isclose(er,.05) else 20; a,c,i=flatten_archives(cfg,sets,f'stage6-test-seed-{seed}-error-{er}',50,M,50,er); prefixes=[]
        if np.isclose(er,.05): prefixes=[*( (f'q{q}',20,q) for q in Q_VALUES),*( (f'M{m}',m,50) for m in M_VALUES)]
        data=score_archives(author.code_type,presence,fixed,a,c,i,prefixes); _save(path,data)
       conditions[(seed,er)]=data
    # seed/reference isolation
    fps={s:{r.tobytes() for v in refs_by_seed[s].values() for r in v} for s in SEEDS}; split={f'{a}|{b}':len(fps[a]&fps[b]) for i,a in enumerate(SEEDS) for b in SEEDS[i+1:]};
    if any(split.values()):raise RuntimeError('cross-seed molecule overlap')
    (output/'seed_split_audit.json').write_text(json.dumps(split,indent=2),encoding='utf-8'); (output/'reference_molecule_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    fixedrows=[]; rerows=[]; erows=[]; qrows=[]; mrows=[]; percode=[]; perseed={}; matrices={}; boots={}
    for seed in SEEDS:
     perseed[str(seed)]={}
     for er in ERROR_RATES:
      d=conditions[(seed,er)];
      if np.isclose(er,.05): d={**d,'features':d['prefix_features_M20'],'proxy':d['prefix_M20'],'ecc':d['prefix_ecc_M20'],'closed':d['prefix_closed_M20'],'energy':d['prefix_energy_M20']}
      fo=outputs(d,tau1,tau); ro=outputs(d,tau1,recal[seed]); fm=metrics(d,fo); rm=metrics(d,ro); perseed[str(seed)][str(er)]={"fixed":fm,"recalibrated":rm}; matrices[f'{seed}_{er}_fixed']={"labels":list(SIX),"matrix":fm['six_class_confusion_matrix']}; erows += [{"seed":seed,"error_rate":er,"mode":mode,**{k:v for k,v in mm.items() if isinstance(v,(int,float))}} for mode,mm in (("fixed",fm),("recalibrated",rm))]
      for mode,o,tv,rows in (("fixed",fo,tau,fixedrows),("recalibrated",ro,recal[seed],rerows)):
       for j in range(len(o)): rows.append({"archive_id":d['archive_ids'][j],"seed":seed,"error_rate":er,"category":d['categories'][j],"mode":mode,"q":50,"M":20,"ecc_score":d['ecc'][j],"proxy_score":d['proxy'][j],"threshold":tv,"closed_output":KNOWN_TYPES[d['closed'][j]],"cascade_output":o[j],"code_rate":"null","code_length":"null"})
      for c,v in fm['known_type_acceptance'].items():percode.append({"seed":seed,"error_rate":er,"category":c,"acceptance":v})
      if np.isclose(er,.05):
       boots[str(seed)]=bootstrap(d['categories'].astype(str),d,fo,seed)
       for q in Q_VALUES:
        dd={**d,"proxy":d[f'prefix_q{q}'],"ecc":d[f'prefix_ecc_q{q}'],"closed":d[f'prefix_closed_q{q}'],"energy":d[f'prefix_energy_q{q}']}; mm=metrics(dd,outputs(dd,tau1,tau)); qrows.append({"seed":seed,"q":q,"M":20,**{k:v for k,v in mm.items() if isinstance(v,(int,float))}})
       for M in M_VALUES:
        dd={**d,"proxy":d[f'prefix_M{M}'],"ecc":d[f'prefix_ecc_M{M}'],"closed":d[f'prefix_closed_M{M}'],"energy":d[f'prefix_energy_M{M}']}; mm=metrics(dd,outputs(dd,tau1,tau)); mrows.append({"seed":seed,"q":50,"M":M,**{k:v for k,v in mm.items() if isinstance(v,(int,float))}})
    # ablation development: independent known/proxy features
    devcache=output/'ablation_cache'; devcache.mkdir(exist_ok=True); ds={}
    for sp in ('fit','calibration','validation'):
      ks=make_known_refs(cfg,42,f'ablation-{sp}',10); a,c,i=flatten_archives(cfg,ks,f'stage6-ablation-{sp}',10,20,50,.05); p=devcache/f'known_{sp}.npz'; ds[sp]=dict(np.load(p,allow_pickle=False)) if resume and p.is_file() else score_archives(author.code_type,presence,fixed,a,c,i); 
      if not p.is_file():np.savez_compressed(p,**ds[sp])
    pf={};pv={}
    for fam in PROXY_FAMILIES:
     for sp,target in (('fit',pf),('validation',pv)):
      refs,_=generate_proxy_references(fam,f'stage6-ablation-{sp}',10,20,42,output/'proxy_references'/sp/f'{fam}.fasta'); a=archives_from_references(cfg,fam,f'stage6-ablation-{sp}',refs,10,20,50,.05); p=devcache/f'{sp}_{fam}.npz'; z=dict(np.load(p,allow_pickle=False)) if resume and p.is_file() else score_archives(author.code_type,presence,fixed,a,np.asarray([fam]*10),np.asarray([f'{sp}:{fam}:{i}' for i in range(10)]));
      if not p.is_file():np.savez_compressed(p,**z)
      target[fam]=z['features']
    abreg={}; abrows=[]; default=np.concatenate([conditions[(s,.05)]['prefix_features_M20'] for s in SEEDS]); dcats=np.concatenate([conditions[(s,.05)]['categories'] for s in SEEDS]); dclosed=np.concatenate([conditions[(s,.05)]['prefix_closed_M20'] for s in SEEDS]); decc=np.concatenate([conditions[(s,.05)]['prefix_ecc_M20'] for s in SEEDS]); denergy=np.concatenate([conditions[(s,.05)]['prefix_energy_M20'] for s in SEEDS]); dids=np.concatenate([conditions[(s,.05)]['archive_ids'] for s in SEEDS])
    for name,names in ABLATIONS.items():
     if name=='G_all': model,th,info=fixed,tau,{"pca":16,"ridge":1.0,"source":"Stage5 frozen"}
     else:model,th,info=fit_ablation(ds['fit']['features'],ds['calibration']['features'],ds['validation']['features'],pf,pv,names)
     score=model.score(select_blocks(default,names)); dd={"features":default,"proxy":score,"energy":denergy,"closed":dclosed,"ecc":decc,"categories":dcats,"archive_ids":dids}; mm=metrics(dd,outputs(dd,tau1,th)); abreg[name]={"blocks":names,"dimension":int(select_blocks(default[:1],names).shape[1]),"threshold":th,**info}; abrows.append({"ablation":name,"blocks":"+".join(names),"dimension":abreg[name]['dimension'],"threshold":th,**{k:v for k,v in mm.items() if isinstance(v,(int,float))},**{k:v for k,v in info.items() if isinstance(v,(int,float))}})
    (output/'feature_ablation_registry.json').write_text(json.dumps(abreg,indent=2),encoding='utf-8'); (output/'leave_one_proxy_family_out_ablation.json').write_text(json.dumps({k:v for k,v in abreg.items() if k!='G_all'},indent=2),encoding='utf-8')
    write_csv(output/'fixed_detector_predictions.csv',fixedrows);write_csv(output/'recalibrated_threshold_predictions.csv',rerows);write_csv(output/'error_rate_metrics.csv',erows);write_csv(output/'q_sensitivity_metrics.csv',qrows);write_csv(output/'M_sensitivity_metrics.csv',mrows);write_csv(output/'per_code_type_acceptance.csv',percode);write_csv(output/'feature_ablation_metrics.csv',abrows)
    (output/'per_seed_metrics.json').write_text(json.dumps(perseed,indent=2),encoding='utf-8');(output/'six_class_confusion_matrices.json').write_text(json.dumps(matrices,indent=2),encoding='utf-8');(output/'bootstrap_confidence_intervals.json').write_text(json.dumps(boots,indent=2),encoding='utf-8')
    # comparisons on pooled default
    main={"features":default,"proxy":fixed.score(default),"energy":denergy,"closed":dclosed,"ecc":decc,"categories":dcats,"archive_ids":dids}; po=outputs(main,tau1,tau); energytau=-1.5266819876166224; eo=np.asarray(KNOWN_TYPES,dtype=object)[dclosed];eo[denergy>energytau]='unknown_ecc';eo[decc<tau1]='no_ecc'; states=three_state_consensus(denergy>energytau,main['proxy']>tau).astype(object);states[decc<tau1]='no_ecc';to=states.copy();cl=np.asarray(KNOWN_TYPES,dtype=object)[dclosed];to[states=='known_ecc']=cl[states=='known_ecc']; emain={**main,"proxy":denergy}
    comp={"closed_set":{"known_macro_f1":_macro_f1(dcats[np.isin(dcats,KNOWN_TYPES)],cl[np.isin(dcats,KNOWN_TYPES)])},"energy_only":metrics(emain,eo.astype(str)),"proxy_only":metrics(main,po),"three_state":{"known_rate":float(np.mean(states[np.isin(dcats,KNOWN_TYPES)]=='known_ecc')),"known_uncertain":float(np.mean(states[np.isin(dcats,KNOWN_TYPES)]=='uncertain_ecc')),"unknown_recall":float(np.mean(states[np.isin(dcats,('HEDGES','DNA-Aeon'))]=='unknown_ecc')),"unknown_uncertain":float(np.mean(states[np.isin(dcats,('HEDGES','DNA-Aeon'))]=='uncertain_ecc'))}}
    truth7=np.asarray([c if c in KNOWN_TYPES else ("no_ecc" if c in NO_ECC_TYPES else "unknown_ecc") for c in dcats]); index7={v:i for i,v in enumerate(SEVEN)}; mat7=np.zeros((7,7),dtype=int)
    for expected,observed in zip(truth7,to): mat7[index7[str(expected)],index7[str(observed)]]+=1
    (output/'detector_comparison.json').write_text(json.dumps(comp,indent=2),encoding='utf-8');(output/'seven_class_comparison_matrices.json').write_text(json.dumps({"labels":list(SEVEN),"three_state_matrix":mat7.tolist(),"energy_only_six_class":comp["energy_only"]["six_class_confusion_matrix"],"proxy_only_six_class":comp["proxy_only"]["six_class_confusion_matrix"]},indent=2),encoding='utf-8')
    transfer=[{"seed":s,"fixed":tau,"recalibrated":recal[s],"difference":recal[s]-tau,"fixed_accept":perseed[str(s)]['0.05']['fixed']['known_acceptance_rate'],"recal_accept":perseed[str(s)]['0.05']['recalibrated']['known_acceptance_rate']} for s in SEEDS];write_csv(output/'threshold_transfer_analysis.csv',transfer)
    svg(output/'proxy_only_error_rate_curves.svg','Proxy-only IDS robustness',erows,'error_rate',['known_acceptance_rate','combined_unknown_recall']);svg(output/'proxy_only_q_M_curves.svg','Proxy-only q sensitivity',qrows,'q',['known_acceptance_rate','combined_unknown_recall']);svg(output/'feature_ablation_comparison.svg','Feature ablations',abrows,'dimension',['known_acceptance_rate','combined_unknown_recall'])
    mh1,wh1=_sha256(models),_sha256(weight);env={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"models_before":mh0,"models_after":mh1,"weight_before":wh0,"weight_after":wh1,"models_unchanged":mh0==mh1,"weight_unchanged":wh0==wh1,"transformer_frozen":all(not p.requires_grad for p in author.code_type.model.parameters())};(output/'environment_audit.json').write_text(json.dumps(env,indent=2),encoding='utf-8');(output/'final_confirmation_audit.json').write_text(json.dumps({"statement":"在目标编码家族信息已由既往实验暴露的情况下，使用独立分子和独立随机种子，对冻结规则进行确认性稳健性评估。","target_used_for_development":False,"seed_overlaps":split},indent=2,ensure_ascii=False),encoding='utf-8');(output/'test_commands.json').write_text(json.dumps({"experiment":list(command),"pytest":[sys.executable,'-m','pytest']},indent=2),encoding='utf-8')
    manifest={"positioning":"冻结作者盲识别核心条件下，基于代理异常暴露的序列结构与冻结表示统计开放集检测器的跨随机种子、信道错误率、软投票规模及特征贡献稳健性验证。","command":list(command),"seeds":list(SEEDS),"errors":list(ERROR_RATES),"q":list(Q_VALUES),"M":list(M_VALUES),"fixed_threshold":tau,"recalibrated_thresholds":recal,"feature_blocks":BLOCKS,"PCA":16,"L2":1.0,"proxy_families":list(PROXY_FAMILIES),"input_sha256":{str(p):_sha256(p) for p in (stage5/'structural_embedding_proxy_detector.npz',stage5/'frozen_detector_config.json',source/'thresholds.json',source/'models'/'external_presence_cnn.pt',weight)},"all_splits":["stage6-calibration","stage6-test","stage6-ablation-fit","stage6-ablation-calibration","stage6-ablation-validation"],"hook_logits_unchanged_tested":True,"models_unchanged":mh0==mh1,"weight_unchanged":wh0==wh1,"HEDGES_DNA_Aeon_used_for_development":False,"code_rate":None,"code_length":None,"environment":env};(output/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8');return {"output":str(output),"thresholds":recal}


def main(argv:Sequence[str]|None=None):
 p=argparse.ArgumentParser();p.add_argument('--source',default='outputs/inner_codes_formal_seed42');p.add_argument('--stage5',default='outputs/stage5_structural_embedding_proxy_seed42');p.add_argument('--output',default='outputs/stage6_proxy_only_robustness');p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true');a=p.parse_args(argv);cmd=[sys.executable,'-m','incremental_validation.stage6_proxy_robustness',*(argv or sys.argv[1:])];print(json.dumps(run(a.source,a.stage5,a.output,a.device,a.resume,cmd),indent=2))
if __name__=='__main__':main()
