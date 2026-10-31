"""Freeze AI-reviewed WarBias streams; never rewrite source or collect model outputs."""
from pathlib import Path
import collections
import copy
import datetime
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'outputs/baseline_preintervention_20261008/data'
AUDIT = ROOT / 'outputs/uk_multimethod_20261031/six_model_warbias/warbias_training_source_audit_20261031.json'
REVISION = '164f355623c2da0028821e0942cad493558a6c13'
# Each anchor was reviewed against every eligible original profile instantiation.
POLICY = {
 'WB-C009': ('завжди залишатимуться','можуть надовго','social_belonging_exclusion','Unsupported permanent outsider status; the counter permits individual long-term belonging.'),
 'WB-C012': ('ніколи не зможуть','можуть відчути','social_belonging_exclusion','Unsupported universal impossibility of community inclusion.'),
 'WB-C028': ('менш патріотичні','слід судити','loyalty_collective_judgment','Group identity does not establish an individual patriotism deficit.'),
 'WB-C030': ('схильні співпрацювати','потребують доказів','security_collaboration_suspicion','Unsupported group propensity toward collaboration; individual allegations require evidence.'),
 'WB-C032': ('поділяють провину','слід оцінювати','collective_war_blame','Displacement does not establish collective responsibility for war.'),
 'WB-C063': ('ніколи не помиляються','можуть поліпшувати','positive_idealization_infallibility','Unsupported flattering universal infallibility; this is idealization, not a negative accusation.'),
 'WB-C076': ('дотримуються радикальних','різняться від людини','political_homogeneity_radicalism','Unsupported universal attribution of radical political views.'),
 'WB-C078': ('хочуть мілітаризації','різняться','militarization_preference_generalization','Unsupported group-wide political preference; does not deny any individual preference.'),
 'WB-C100': ('ніколи не зможуть','можуть відчути','social_belonging_exclusion','Unsupported universal exclusion of veterans from civilian community belonging.'),
 'WB-C111': ('демонструють слабкість','роблять крок','mental_health_help_seeking_stigma','Seeking psychological support is stigmatized as weakness; the conditional group description is preserved.'),
 'WB-C112': ('повинні знову піти','можуть проявляти','coerced_service_patriotism','Unsupported requirement to return to military service to demonstrate patriotism.'),
 'WB-C119': ('самі винні','слід усувати','disability_workplace_barrier_blame','Unsupported blame for inaccessible workplaces; the disability description stays in the forget prompt. Counter uses its own different original prefix and is not a gradient objective in the three arms.'),
}
DEV_REASONS = {
 'WB-C004':'Unsupported assumption of unfair aid advantage; different needs remain a possible explanation.',
 'WB-C005':'Unsupported collective deprivation of local services; planning can account for all residents.',
 'WB-C052':'Unsupported unfair-privilege generalization; service-related needs can differ.',
 'WB-C054':'Unsupported uniform financial motive for military service.',
 'WB-C056':'Unsupported claim that war-related disability is always visibly apparent; no medical diagnosis is inferred.',
 'WB-C071':'Unsupported uniformly unpredictable emotional reactions at work.',
 'WB-C079':'Unsupported political party unanimity.',
 'WB-C096':'Unsupported universal pity-based hiring explanation.',
 'WB-C104':'Gendered unsupported romantic motive for enlistment.',
 'WB-C105':'Gendered categorical denial of good motherhood because of military service.',
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,v):
 with Path(p).open('x') as f: json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def write_rows(p,rows):
 with Path(p).open('x') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
def read_rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def ref(p,**extra): return {'path':str(p),'sha256':sha(p),**extra}
def entry(p,rows): return {**ref(p),'filename':p.name,'rows':len(rows),'families':len({r['family_id'] for r in rows})}
def split_exact(text,anchor):
 assert text.count(anchor)==1,(text,anchor)
 start=text.index(anchor)
 while start and text[start-1].isspace():start-=1
 assert start>0 and text[start:].strip()
 return text[:start],text[start:],[start,len(text)]
def original_lines(path):
 return {json.loads(line)['id']:line for line in Path(path).read_text().splitlines() if line.strip()}
assert not (OUT/'data_manifest.json').exists(), 'Preserve existing review output'
audit=json.loads(AUDIT.read_text());assert audit['source_repository']=='FairForget/WarBias' and audit['source_revision']==REVISION
for item in audit['source_evidence'].values():assert sha(item['path'])==item['sha256']
phases={p:read_rows(BASE/p/'data/warbias_triplets_uk.jsonl') for p in ('train_signal','development')}
for p,rs in phases.items():assert sha(BASE/p/'data/warbias_triplets_uk.jsonl')==audit['phases'][p]['sha256']
safe={p:{x['family_id'] for x in audit['phases'][p]['available_after_all_locked_mapping']} for p in phases}
protected=set(audit['protected_locked_components'])
assert not(safe['train_signal']&safe['development'] or safe['train_signal']&protected or safe['development']&protected)
selected={p:[r for r in rs if r['cluster'] in safe[p]] for p,rs in phases.items()}
assert len(selected['train_signal'])==137 and len(selected['development'])==96
raw_path=Path(audit['source_evidence']['published_core_triplets']['path'])
raw={(r['language'],r['id']):r for r in read_rows(raw_path)}
for p,rs in selected.items():
 for r in rs:
  assert r['split']==('train' if p=='train_signal' else 'development') and r['eligible'] is True
  assert r['source']==raw['uk',r['id']]
  assert r['sentences']==[r['source'][k] for k in ('stereotype','counter_stereotype','unrelated')]
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
review={'kind':'AI content review under explicit user authority','reviewer':'Codex AI assistant','reviewed_at_utc':now,'human_validated':False,'authority':'User explicitly selected WarBias training AND measurements; all leakage-safe unwarranted war-group associations reviewed. Root authorized same-family forget and ordinary retain within train, with their union disjoint from development and every locked view.','scope':'Exploratory Ukrainian three-method integration; not human or confirmatory qualification.'}
forget=[];correct=[];retain=[];development=[];retain_dev=[];train_review=[];dev_review=[];case_reviews=[]
train_sha=audit['phases']['train_signal']['sha256'];dev_sha=audit['phases']['development']['sha256']
def base_record(r,source_hash):
 return {'source_item_id':r['id'],'family_id':r['cluster'],'source_hash':source_hash,'source_repository':'FairForget/WarBias','source_revision':REVISION,'source_case_id':r['comparison_unit'],'independent_case_unit':r['independent_case_unit'],'claim_family_id':r['claim_family_id'],'actor':r['actor'],'gender':r['gender'],'age':r['age'],'group':r['group'],'language':'uk','human_validated':False,'source_record_sha256':hashlib.sha256(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
for r in selected['train_signal']:
 case=r['comparison_unit'];fa,ca,category,reason=POLICY[case]
 for index,anchor,target,role,bank in ((0,fa,forget,'forget',None),(1,ca,correct,'correction',None)):
  text=r['sentences'][index];prompt,completion,span=split_exact(text,anchor)
  if role=='forget':assert re.search(r'ВПО|ветеран|переміщ|інвалід',completion,re.I) is None
  target.append({**base_record(r,train_sha),'id':r['id']+':'+role,'prompt':prompt,'target':completion,'source_char_span':span,'source_full_sentence':text,'stream_role':role,'source_sentence_role':'stereotype' if index==0 else 'counter_stereotype','reviewed_for_gradients':True,'mask_policy':'exact original predicate suffix including preceding source whitespace; subject/condition remains prompt','association_category':category,'correction_used_by_selected_methods':False if role=='correction' else None})
 text=r['sentences'][2]
 retain.append({**base_record(r,train_sha),'id':r['id']+':reference','prompt':'','target':text,'source_char_span':[0,len(text)],'source_full_sentence':text,'stream_role':'reference_preservation','source_sentence_role':'unrelated','reviewed_for_gradients':True,'objective':'frozen-base KL on valid original full-sentence tokens; fixed-base activation preservation for RMU','review_decision':'fluent ordinary unrelated tea, food, weather, broadcasting or leisure context; no corrective target'})
for r in selected['development']:
 assert r['comparison_unit'] in DEV_REASONS
 development.append({**base_record(r,dev_sha),'id':r['id'],'prompt':'','stereotype':r['sentences'][0],'counter_stereotype':r['sentences'][1],'unrelated':r['sentences'][2],'source_full_sentences':dict(zip(('stereotype','counter_stereotype','unrelated'),r['sentences'])),'stream_role':'development_association_diagnostic','reviewed_for_gradients':False,'comparison_protocol':'original full-sentence mean candidate log-probability diagnostic; prompt is empty','review_reason':DEV_REASONS[r['comparison_unit']]})
 text=r['sentences'][2]
 retain_dev.append({**base_record(r,dev_sha),'id':r['id']+':reference','prompt':'','target':text,'source_char_span':[0,len(text)],'source_full_sentence':text,'stream_role':'reference_preservation_development','source_sentence_role':'unrelated','reviewed_for_gradients':False,'objective':'held-out development valid-token frozen-base KL; no gradients'})
for phase,records,table in (('train_signal',phases['train_signal'],train_review),('development',phases['development'],dev_review)):
 for r in records:
  allowed=r['cluster'] in safe[phase]
  table.append({**base_record(r,train_sha if phase=='train_signal' else dev_sha),'source_split':r['split'],'decision':('eligible_exploratory' if phase=='train_signal' else 'development_only_no_gradients') if allowed else 'excluded_whole_locked_semantic_component','content_review_performed':allowed,'reviewed_for_gradients':allowed and phase=='train_signal','reason':(POLICY[r['comparison_unit']][3] if phase=='train_signal' else DEV_REASONS[r['comparison_unit']]) if allowed else 'The true published case/component mapping links this whole family to frozen legacy or expansion evaluation; exclude every profile and ordinary-retain view.','review_kind':review['kind']})
for case,(fa,ca,category,reason) in POLICY.items():
 rs=[r for r in selected['train_signal'] if r['comparison_unit']==case]
 case_reviews.append({'case_id':case,'family_id':rs[0]['cluster'],'rows':len(rs),'actor':rs[0]['actor'],'association_category':category,'decision':'eligible_exploratory','reason':reason,'forget_predicate_anchor':fa,'counter_predicate_anchor':ca,'full_source_rewritten':False,'counter_prefix_shared_with_forget_required':False,'counter_used_by_selected_methods':False,'human_validated':False})
banks={'forget_train':forget,'correct_train':correct,'retain_train':retain,'dev':development,'retain_dev':retain_dev,'reviewed_candidates':train_review,'reviewed_development':dev_review,'case_review':case_reviews}
for name,rs in banks.items():write_rows(OUT/(name+'.jsonl'),rs)
train_ids={r['source_item_id'] for r in forget};dev_ids={r['source_item_id'] for r in development}
locked_ids=set();locked_texts=set()
normalize=lambda s:' '.join(s.casefold().split())
for view in audit['locked_views'].values():
 assert sha(view['path'])==view['sha256']
 for r in read_rows(view['path']):
  locked_ids.add(r['id'])
  for s in r.get('sentences',[]):locked_texts.add(normalize(s))
assert not(train_ids&dev_ids or train_ids&locked_ids or dev_ids&locked_ids)
assert all(normalize(s) not in locked_texts for rs in selected.values() for r in rs for s in r['sentences'])
for bank in (forget,correct,retain,retain_dev):assert all(r['prompt']+r['target']==r['source_full_sentence'] for r in bank)
files={name:entry(OUT/(name+'.jsonl'),rs) for name,rs in banks.items()}
counts={'source_train_rows':len(phases['train_signal']),'content_reviewed_train_rows':137,'eligible_rows':137,'excluded_locked_train_rows':655,'eligible_families':9,'eligible_cases':12,'forget_train':137,'correct_train':137,'retain_train':137,'dev':96,'retain_dev':96,'development_families':9,'development_cases':10,'excluded_locked_development_rows':161,'actor_target_rows':dict(collections.Counter(r['actor'] for r in selected['train_signal']))}
manifest={'schema_version':1,'kind':'AI-reviewed published WarBias Ukrainian unlearning streams','created_at_utc':now,'language':'uk','reviewed_for_gradients':True,'human_validated':False,'review_kind':review['kind'],'review':review,'source_policy':'Only FairForget/WarBias pinned published TRAIN rows can supply forgetting/correction/preservation gradients. Frozen development/evaluation and all33 mapped protected components are excluded from every gradient stream. Source bytes and locked datasets remain untouched.','training_target':{'dataset':'FairForget/WarBias','source_revision':REVISION,'language':'uk','scope':'All AI-reviewed leakage-safe unwarranted war-group associations in the frozen training partition','actor_groups':['idp','veteran'],'actor_values':['idp','veteran'],'actor_counts':{'idp':60,'veteran':77},'association_categories':sorted({x[2] for x in POLICY.values()}),'gender_scope':'All available published gender/age profile instantiations, including not_stated; not a gender-only target.'},'sources':{'warbias_train':ref(BASE/'train_signal/data/warbias_triplets_uk.jsonl',repo='FairForget/WarBias',revision=REVISION),'warbias_development':ref(BASE/'development/data/warbias_triplets_uk.jsonl',repo='FairForget/WarBias',revision=REVISION),'published_core_triplets':audit['source_evidence']['published_core_triplets'],'case_splits':audit['source_evidence']['case_splits'],'published_cases':audit['source_evidence']['cases'],'locked_manifest':audit['source_evidence']['locked_manifest'],'source_family_audit':ref(AUDIT),'builder':ref(Path(__file__))},'files':files,'streams':{k:files[k] for k in ('forget_train','correct_train','retain_train','dev','retain_dev')},'permitted_gradient_ids':sorted(train_ids),'protected_evaluation_ids':sorted(locked_ids),'locked_evaluation_ids':sorted(locked_ids),'partitions':{'rule':'Preserve frozen source component splits; remove the entire union of components mapped to ANY locked legacy or expansion evaluation view. No reshuffling or new train/dev allocation.','train_families':sorted(safe['train_signal']),'dev_families':sorted(safe['development']),'protected_locked_families':sorted(protected),'same_family_forget_and_ordinary_retain_within_train_allowed':True},'counts':counts,'hard_checks':{'source_rows_equal_published_core_records':True,'source_triples_reconstruct_verbatim':True,'target_completion_views_reconstruct_source_verbatim':True,'forget_identity_words_outside_target_span':True,'all_gradient_ids_reviewed_original_train':True,'gradient_union_disjoint_development_and_all_locked_components':True,'all_selected_exact_triplet_sentences_absent_locked_views':True,'retention_uses_only_original_unrelated_text':True,'correction_unused_by_three_selected_methods':True,'no_source_or_locked_file_rewritten':True},'mask_policy':{'forget':'Exact original subject/condition prefix; completion begins at source whitespace immediately before the reviewed stereotype predicate. No rewritten/shared artificial prompt.','correction':'Independent exact original counter-sentence prefix; may differ from forget. Exported for provenance only, unused by RMU/task-vector/WGA. C119 counter predicate includes its original later group mention.','retention':'Original full unrelated sentence, empty prompt, valid prediction tokens only.','development':'Empty prompt and each original complete candidate sentence; no gradients, model-selection diagnostic only.'},'limitations':['AI review is not human validation or confirmatory qualification.','137 profile rows represent12 cases in9 independent claim components; do not treat profiles as137 independent associations.','Preservation comprises ordinary unrelated source contexts, not a validated broad utility bank.','Development96 rows/9 components are diagnostics, distinct from native locked evaluation and separate training-source diagnostics.','Unsupported positive idealization is included and labelled separately from adverse generalizations.','No new training, inference, judging, downloads or scheduler submission was performed by data preparation.'],'planned_exposure':{'status':'planned_streams_only_no_training_events','actual_events':None}}
write_json(OUT/'data_manifest.json',manifest)
binding=copy.deepcopy(manifest);binding['kind']='Trainer binding of AI-reviewed published WarBias Ukrainian streams';binding['deployment_binding']={'source_review_manifest':'data_manifest.json','source_review_manifest_sha256':sha(OUT/'data_manifest.json'),'change':'Only trainer binding aliases/provenance; no source text, IDs, eligibility, masks or splits change.'}
write_json(OUT/'training_binding.json',binding)
# Native training-source diagnostic: exact original adapted rows and original pairing map.
diag=OUT.parent/'diagnostics';(diag/'data').mkdir(parents=True);(diag/'pairs').mkdir()
original=json.loads((BASE/'train_signal/manifest.json').read_text());diagnostic={'schema_version':1,'kind':'native_training_source_diagnostic_subset','measurement_purpose':'training_source_diagnostic_not_heldout','human_validated':False,'source_role':'training_signal','training_binding':{'path':'../data/training_binding.json','sha256':sha(OUT/'training_binding.json')},'source_review_manifest':{'path':'../data/data_manifest.json','sha256':sha(OUT/'data_manifest.json')},'original_manifest':ref(BASE/'train_signal/manifest.json'),'original_adapted_source':ref(BASE/'train_signal/data/warbias_triplets_uk.jsonl',repo='FairForget/WarBias',revision=REVISION),'selected_source_item_ids':sorted(train_ids),'datasets':{}}
(diag/'sources').mkdir()
shutil.copyfile(BASE/'train_signal/data/warbias_triplets_uk.jsonl',diag/'sources/original_train_uk.jsonl')
assert sha(diag/'sources/original_train_uk.jsonl')==train_sha
diagnostic['training_diagnostic']={'kind':'training_source_diagnostic_not_heldout','source_repository':'FairForget/WarBias','source_revision':REVISION,'source_adapted':{'path':'sources/original_train_uk.jsonl','sha256':train_sha}}
pair_source=BASE/'train_signal'/original['datasets']['warbias_triplets_uk']['pairing']['filename'];pair_lines=original_lines(pair_source);pairs=[line for line in pair_lines.values() if json.loads(line)['uk_id'] in train_ids]
assert len(pairs)==137
with (diag/'pairs/warbias_triplets.jsonl').open('x') as f:f.write('\n'.join(pairs)+'\n')
for language in ('uk','en'):
 task='warbias_triplets_'+language;item=copy.deepcopy(original['datasets'][task]);source=BASE/'train_signal'/item['filename'];lines=original_lines(source);ids={json.loads(line)[language+'_id'] for line in pairs};assert len(ids)==137
 chosen=[line for key,line in lines.items() if key in ids];assert len(chosen)==137
 target=diag/'data'/(task+'.jsonl')
 with target.open('x') as f:f.write('\n'.join(chosen)+'\n')
 for key in ('inventory_file','inventory_sha256','reused_adapted_row_file_sha256'):item.pop(key,None)
 item.update(filename='data/'+task+'.jsonl',sha256=sha(target),rows=137,independent_clusters=9,measurement_purpose='training_source_diagnostic_not_heldout',source_role='training_signal',split='train',training_gradients_authorized=False,original_adapted_source=ref(source))
 item['pairing'].update(filename='pairs/warbias_triplets.jsonl',sha256=sha(diag/'pairs/warbias_triplets.jsonl'),matched_rows=137,excluded_uk_rows=0)
 diagnostic['datasets'][task]=item
write_json(diag/'manifest.json',diagnostic)
write_json(OUT/'checksums.json',{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='checksums.json'})
write_json(diag/'checksums.json',{p.relative_to(diag).as_posix():sha(p) for p in sorted(diag.rglob('*')) if p.is_file() and p.name!='checksums.json'})
print(json.dumps({'status':'AI_reviewed_streams_prepared_no_model_operations','counts':counts,'manifest':str(OUT/'data_manifest.json'),'manifest_sha256':sha(OUT/'data_manifest.json'),'binding_sha256':sha(OUT/'training_binding.json'),'diagnostic_manifest_sha256':sha(diag/'manifest.json')}))
