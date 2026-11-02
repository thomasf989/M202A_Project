'use strict';
const $ = id => document.getElementById(id);
const state = {file:null, ready:false, busy:false, scene:null, drawn:false, loadingScene:false, result:null, objectUrl:null, cameraModeInitialized:false, currentRoom:null, managing:false, uploading:false, pendingRoom:null, statusTimer:null, statusEpoch:0};
const cameraDefault = {up:{x:0,y:1,z:0}, eye:{x:1.65,y:1.1,z:1.65}, center:{x:0,y:0,z:0}};
const baseLayout = {
  paper_bgcolor:'#f8faf6', plot_bgcolor:'#f8faf6', margin:{l:0,r:0,t:0,b:0}, showlegend:false,
  uirevision:'keep-room-view', scene:{aspectmode:'data', bgcolor:'#f8faf6', camera:cameraDefault,
    xaxis:{title:'X · meters',color:'#9aa894',gridcolor:'#e0e8db',zeroline:false,showbackground:false},
    yaxis:{title:'Y · meters',color:'#9aa894',gridcolor:'#e0e8db',zeroline:false,showbackground:false},
    zaxis:{title:'Z · meters',color:'#9aa894',gridcolor:'#e0e8db',zeroline:false,showbackground:false}}
};
function error(message){$('errorMessage').textContent=message;$('errorMessage').hidden=!message;}
function controls(){
  $('localizeButton').disabled=!state.ready||!state.file||state.busy;
  $('sampleButton').disabled=!state.ready||state.busy;
  $('dropzone').disabled=state.busy;$('photoInput').disabled=state.busy;
  $('cameraMode').disabled=state.busy;$('importCalibration').disabled=state.busy;
  $('localizeLabel').textContent=state.busy?'Localizing…':'Localize This Photo';
  $('roomSelect').disabled=state.busy||state.uploading;$('addRoom').disabled=state.busy||state.uploading;
}
async function api(url, options={}){
  const response=await fetch(url,options);
  const data=await response.json();
  if(!response.ok)throw new Error(data.error||'Processing failed. Please try again.');
  return data;
}
function progress(text,value){
  $('progressWrap').hidden=false;$('progressText').textContent=text;
  $('progressBar').style.width=`${Math.max(3,Math.min(100,value*100))}%`;
}
async function drawScene(){
  if(!state.scene)return;
  const traces=state.scene.data.map(t=>({...t}));
  traces[0].opacity=Number($('opacity').value);traces[1].visible=$('showReferences').checked;
  const colors=traces[0].meta?.color_modes?.[$('colorMode').value];
  if(colors)traces[0].vertexcolor=colors;
  if(state.result?.success)traces.push(...cameraTraces(state.result.pose));
  const layout=structuredClone(baseLayout);
  if(state.drawn&&$('scene').layout?.scene?.camera)layout.scene.camera=$('scene').layout.scene.camera;
  await Plotly.react('scene',traces,layout,{responsive:true,displaylogo:false,scrollZoom:true,
    modeBarButtonsToRemove:['toImage','select2d','lasso2d'],displayModeBar:false});
  state.drawn=true;$('sceneLoading').hidden=true;
}
function cameraTraces(pose){
  const T=pose.T_world_camera_cv, origin=pose.position_m;
  const transform=v=>T.slice(0,3).map(row=>row[0]*v[0]+row[1]*v[1]+row[2]*v[2]+row[3]);
  const corners=[[-.13,-.10,.35],[.13,-.10,.35],[.13,.10,.35],[-.13,.10,.35]].map(transform);
  const lines=[];
  for(let i=0;i<4;i++){lines.push(origin,corners[i],null,corners[i],corners[(i+1)%4],null);}
  const forward=transform([0,0,.48]);lines.push(origin,forward,null);
  return [
    {type:'scatter3d',x:lines.map(p=>p?.[0]??null),y:lines.map(p=>p?.[1]??null),z:lines.map(p=>p?.[2]??null),
      mode:'lines',line:{color:'#ed4943',width:6},hoverinfo:'skip',name:'Camera direction'},
    {type:'scatter3d',x:[origin[0]],y:[origin[1]],z:[origin[2]],mode:'markers+text',text:['Camera position'],
      textposition:'top center',textfont:{size:13,color:'#cf433c'},marker:{size:7,color:'#ed4943',line:{width:2,color:'white'}},
      hovertemplate:'Camera position<br>X: %{x:.3f} m<br>Y: %{y:.3f} m<br>Z: %{z:.3f} m<extra></extra>'}
  ];
}
function resetRoom(id){
  state.currentRoom=id;state.scene=null;state.result=null;state.drawn=false;state.file=null;state.ready=false;
  state.cameraModeInitialized=false;state.loadingScene=false;
  if(state.objectUrl)URL.revokeObjectURL(state.objectUrl);state.objectUrl=null;
  if(window.Plotly)Plotly.purge('scene');
  $('resultCard').hidden=true;$('focusView').disabled=true;$('sceneLoading').hidden=false;
  $('photoPreview').hidden=true;$('dropPrompt').hidden=false;$('changePhoto').hidden=true;
  $('photoInput').value='';$('filename').textContent='No photo selected';$('fileSize').textContent='';
  $('calibrationStatus').textContent='';$('cameraMode').value='auto';error('');controls();
}
async function loadRooms(){
  try{
    const value=await api('/api/rooms');state.managing=true;$('roomToolbar').hidden=false;
    const rows=value.rooms.map(r=>({id:r.id,label:`${r.name}${r.ready?'':r.phase==='error'?' · Preparation failed':r.phase==='importing'||r.phase==='building'?' · Preparing':''}`}));
    const key=JSON.stringify(rows);
    if($('roomSelect').dataset.key!==key){
      $('roomSelect').replaceChildren(...(rows.length?rows.map(r=>new Option(r.label,r.id)):[new Option('No rooms yet','')]));
      $('roomSelect').dataset.key=key;
    }
    $('roomSelect').value=value.active||'';
  }catch(e){if(!state.managing)$('roomToolbar').hidden=true;}
}
async function status(){
  clearTimeout(state.statusTimer);const epoch=state.statusEpoch;
  try{
    const value=await api('/api/status');if(epoch!==state.statusEpoch)return;
    const id=value.room_id|| (value.phase==='empty'?'empty':'single');
    if(state.currentRoom!==id)resetRoom(id);
    state.ready=value.ready;
    const empty=value.phase==='empty';
    $('roomStatus').textContent=value.ready?'Room ready':empty?'Add your first room':value.phase==='error'?'Room preparation failed':'Preparing room';
    $('roomDetail').textContent=value.ready?`${value.references} reference photos · Ready to localize`:value.message;
    $('statusDot').className=`status-dot ${value.ready?'ready':value.phase==='error'?'error':''}`;
    $('savedOption').disabled=!value.has_calibration;
    const seven=value.dataset_format==='7-scenes';$('datasetOption').disabled=!seven;
    $('roomName').textContent=value.scene_name||'Your room';
    $('sampleButton').textContent=seven&&value.test_frames?'Try Next Test Photo →':'Try a Sample Photo →';
    if(seven&&value.ready)$('roomDetail').textContent=`${value.references} reference photos · ${value.test_frames} test photos`;
    if(value.has_calibration)$('calibrationStatus').textContent=`Saved parameters: ${value.calibration_size.join(' × ')} pixels`;
    if(!state.cameraModeInitialized&&value.ready){
      $('cameraMode').value=seven?'dataset':value.has_calibration?'saved':'auto';
      $('cameraMode').dispatchEvent(new Event('change'));state.cameraModeInitialized=true;
    }
    controls();
    $('sceneLoading').querySelector('b').textContent=empty?'Create a room reference library first':'Loading room model';
    $('sceneLoading').querySelector('small').textContent=empty?'Click “Add Room” above to upload a GLB and reference data':'You can start localizing once the room is ready';
    $('sceneLoading').querySelector('.spinner').hidden=empty;
    if(!empty&&!state.scene&&!state.loadingScene){
      state.loadingScene=true;const room=state.currentRoom;
      try{const scene=await api('/api/scene');if(room===state.currentRoom){state.scene=scene;await drawScene();}}
      catch(e){if(value.ready){$('sceneLoading').querySelector('b').textContent='3D view is temporarily unavailable';$('sceneLoading').querySelector('small').textContent='You can still localize and download the GLB; use a browser that supports WebGL';}}
      finally{state.loadingScene=false;}
    }
    if(!value.ready&&!empty&&value.phase!=='error')progress(value.message,value.progress||0);
    else if(value.phase==='error'){error(value.message);$('progressWrap').hidden=true;}
    else if(!state.busy)$('progressWrap').hidden=true;
    if(state.managing)await loadRooms();
  }catch(e){error('Could not connect to the program. Keep the terminal running and refresh the page.');state.ready=false;controls();}
  finally{state.statusTimer=setTimeout(status,1600);}
}

function showPhoto(url){
  $('photoPreview').src=url;$('photoPreview').hidden=false;$('dropPrompt').hidden=true;$('changePhoto').hidden=false;
}
async function selectFile(file){
  if(!file||state.busy)return;
  if(file.size>40*1024*1024){error('The photo exceeds 40 MB. Export it as a smaller JPG first.');return;}
  if(!/\.(jpe?g|png|heic|heif)$/i.test(file.name)){error('Select a JPG, PNG, or HEIC photo.');return;}
  state.file=file;state.result=null;
  if(state.objectUrl)URL.revokeObjectURL(state.objectUrl);
  state.objectUrl=URL.createObjectURL(file);showPhoto(state.objectUrl);
  $('filename').textContent=file.name;$('fileSize').textContent=`${(file.size/1024/1024).toFixed(1)} MB`;
  $('resultCard').hidden=true;$('focusView').disabled=true;$('sceneHint').hidden=false;
  $('sceneHint').textContent='Click “Localize This Photo” to show the camera position in the model';
  error('');controls();await drawScene();
}
$('photoPreview').addEventListener('error',()=>{
  $('photoPreview').hidden=true;$('dropPrompt').hidden=false;
  $('dropPrompt').querySelector('strong').textContent='Photo selected';
  $('dropPrompt').querySelector('small').textContent='A photo preview will be generated automatically during localization';
});
$('dropzone').addEventListener('click',()=>$('photoInput').click());
$('photoInput').addEventListener('change',event=>selectFile(event.target.files[0]));
for(const type of ['dragenter','dragover'])$('dropzone').addEventListener(type,event=>{
  event.preventDefault();if(!state.busy)$('dropzone').classList.add('dragging');
});
for(const type of ['dragleave','drop'])$('dropzone').addEventListener(type,event=>{
  event.preventDefault();$('dropzone').classList.remove('dragging');
});
$('dropzone').addEventListener('drop',event=>selectFile(event.dataTransfer.files[0]));
// Keep a file dropped just outside the target from navigating away from the app.
window.addEventListener('dragover',event=>event.preventDefault());
window.addEventListener('drop',event=>event.preventDefault());
async function begin(sample=false){
  if(state.busy||!state.ready||(!sample&&!state.file))return;
  state.busy=true;state.result=null;error('');controls();
  $('resultCard').hidden=true;$('focusView').disabled=true;progress('Submitting photo…',.02);
  try{
    let job;
    if(sample){
      state.file=null;controls();
      $('filename').textContent='Sample photo';$('fileSize').textContent=$('cameraMode').value==='dataset'?'Independent test sequence':'Sample from scan';
      job=await api('/api/sample',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    }else{
      const form=new FormData();form.append('image',state.file,state.file.name);form.append('mode',$('cameraMode').value);form.append('solver_mode',$('solverMode').value);
      const baseline=$('multisetPoseInput')?.files?.[0];if(baseline)form.append('multiset_pose',baseline,baseline.name);
      job=await api('/api/localize',{method:'POST',body:form});
    }
    await drawScene();poll(job.id);
  }catch(e){state.busy=false;controls();error(e.message);$('progressWrap').hidden=true;}
}
async function poll(id){
  try{
    const job=await api(`/api/jobs/${id}`);progress(job.message,job.progress);
    if(job.preview_url)showPhoto(job.preview_url);
    if(job.status==='done'){
      state.busy=false;state.result=job.result;controls();$('progressWrap').hidden=true;
      $('filename').textContent=job.filename;showResult(job.result);await drawScene();
    }else if(job.status==='error'){
      state.busy=false;controls();$('progressWrap').hidden=true;error(job.message);
    }else setTimeout(()=>poll(id),600);
  }catch(e){state.busy=false;controls();error(e.message);$('progressWrap').hidden=true;}
}
function showResult(result){
  $('resultCard').hidden=false;$('successContent').hidden=!result.success;$('failureContent').hidden=result.success;
  $('resultTitle').textContent=result.success?'Camera Position Found':'Unable to Localize';
  $('successIcon').hidden=!result.success;$('sceneHint').hidden=result.success;
  $('focusView').disabled=!result.success;
  if(!result.success){
    const reasons={insufficient_geometric_support:'Too few matching points or insufficient image distribution',pnp_ransac_failed:'The geometric position calculation did not pass validation',
      too_few_2d_3d_correspondences:'Too few valid 3D matching points',no_valid_depth_matches:'No usable depth matches',
      no_geometrically_valid_depth_matches:'Candidate reference images have visual matches, but geometric validation or 3D depth produced no reliable matches',
      too_few_query_features_or_references:'The photo has too few texture features, or the reference library is insufficient',no_eligible_references:'No eligible reference viewpoints'};
    let detail=reasons[result.reason]||result.reason||'Insufficient conditions for localization';
    if(result.inliers!==undefined)detail+=` · ${result.inliers} geometric inliers`;
    if(result.correspondences!==undefined)detail+=` / ${result.correspondences} 2D–3D candidates`;
    if(result.enhancement_help)detail+=`。${result.enhancement_help}`;
    $('failureReason').textContent=detail;
    $('sceneHint').textContent='No estimated position is shown. Try another viewpoint.';return;
  }
  result.pose.position_m.forEach((v,i)=>$(['posX','posY','posZ'][i]).textContent=v.toFixed(3));
  let quality=`${result.inliers} reliable matches · ${result.total_seconds.toFixed(1)} seconds`;
  if(result.method)quality+=` · ${result.method}`;
  if(result.pose_confidence!==undefined)quality+=` · Geometric confidence ${Number(result.pose_confidence).toFixed(2)}`;
  if(result.retrieval_method)quality+=` · ${result.retrieval_method}`;
  $('resultQuality').textContent=quality;
  if(result.position_error_m!==undefined)$('resultQuality').textContent+=` · Position error ${(result.position_error_m*100).toFixed(1)} cm · Orientation error ${result.rotation_error_deg.toFixed(2)}°`;
  $('cameraNote').textContent=result.camera.note;$('cameraNote').hidden=!result.camera.estimated&&!result.sample;
  const ms=$('multisetComparison');
  if(result.multiset_position_difference_m!==undefined){
    let text=`Position difference from the MultiSet baseline: ${(result.multiset_position_difference_m*100).toFixed(1)} cm`;
    if(result.multiset_rotation_difference_deg!==undefined)text+=` · Orientation difference ${result.multiset_rotation_difference_deg.toFixed(2)}°`;
    if(result.multiset_baseline?.confidence!==undefined&&result.multiset_baseline.confidence!==null)text+=` · MultiSet confidence ${Number(result.multiset_baseline.confidence).toFixed(2)}`;
    ms.textContent=text+' (baseline, not ground truth)';ms.hidden=false;
  }else{ms.hidden=true;ms.textContent='';}
  $('downloadModel').href=result.urls.model;$('downloadPose').href=result.urls.pose;$('openViewer').href=result.urls.viewer;
  if(window.innerWidth<760)$('resultCard').scrollIntoView({behavior:'smooth',block:'nearest'});
}
$('localizeButton').addEventListener('click',()=>begin(false));
$('sampleButton').addEventListener('click',()=>begin(true));
$('importCalibration').addEventListener('click',()=>$('calibrationInput').click());
$('calibrationInput').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  try{
    const value=JSON.parse(await file.text());
    const saved=await api('/api/calibration',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
    $('savedOption').disabled=false;$('cameraMode').value='saved';
    $('calibrationStatus').textContent=`Parameters saved: ${saved.width} × ${saved.height} pixels`;
    $('cameraMode').dispatchEvent(new Event('change'));error('');
  }catch(e){error(`Parameter import failed: ${e.message}`);}
});
$('cameraMode').addEventListener('change',()=>{
  const saved=$('cameraMode').value==='saved';
  const dataset=$('cameraMode').value==='dataset';
  $('cameraModeLabel').textContent=dataset?'7-Scenes parameters':saved?'Saved parameters':'Automatic estimation';
  if(dataset){$('cameraHelp').textContent='For original 640×480 7-Scenes photos. Uses approximate RGB intrinsics; test photos are not added to the reference library.';return;}
  $('cameraHelp').textContent=saved?'Reuse only with the same lens, crop, and orientation. Import matching parameters after changing the lens or capture mode.':
    'Focal-length information in the photo is used when available. Automatic-mode parameters are estimates; importing calibration parameters can improve reliability.';
});
$('showReferences').addEventListener('change',drawScene);
$('colorMode').addEventListener('change',drawScene);
$('opacity').addEventListener('input',()=>{if(state.drawn)Plotly.restyle('scene',{opacity:Number($('opacity').value)},[0]);});
$('resetView').addEventListener('click',()=>{if(state.drawn)Plotly.relayout('scene',{'scene.camera':structuredClone(cameraDefault)});});
$('focusView').addEventListener('click',()=>{
  if(!state.result?.success)return;
  const p=state.result.pose.position_m,bounds=state.scene.layout.meta.bounds;
  const spans=bounds[1].map((v,i)=>v-bounds[0][i]);const maxSpan=Math.max(...spans);
  const center=p.map((v,i)=>(v-(bounds[0][i]+bounds[1][i])/2)/maxSpan);
  Plotly.relayout('scene',{'scene.camera':{up:{x:0,y:1,z:0},center:{x:center[0],y:center[1],z:center[2]},eye:{x:.8,y:.55,z:.8}}});
});
loadRooms().then(()=>{status();try{const pending=localStorage.getItem('room-build');if(pending){state.pendingRoom=pending;watchBuild(pending);}}catch(e){}});
function roomProgress(message,fraction){
  $('roomUploadProgress').hidden=false;$('roomUploadText').textContent=message;
  $('roomUploadBar').style.width=`${Math.max(2,Math.min(100,fraction*100))}%`;
}
function roomError(message){$('roomUploadError').textContent=message;$('roomUploadError').hidden=!message;}
function uploadControls(){
  const disabled=state.uploading||Boolean(state.pendingRoom);
  for(const id of ['prepareRoomButton','roomModelInput','roomRefsInput','newRoomName','roomFormat','trainStride','testStride','sevenCoordinates','worldTransform'])$(id).disabled=disabled;
  $('closeRoomDialog').disabled=state.uploading;
  controls();
}
$('addRoom').addEventListener('click',()=>{
  if(!state.pendingRoom){roomError('');$('roomUploadProgress').hidden=true;}
  $('roomDialog').showModal();uploadControls();
});
$('closeRoomDialog').addEventListener('click',()=>$('roomDialog').close());
$('roomDialog').addEventListener('cancel',event=>{if(state.uploading)event.preventDefault();});
for(const [input,label] of [['roomModelInput','roomModelName'],['roomRefsInput','roomRefsName']]){
  $(input).addEventListener('change',()=>{$(label).textContent=$(input).files[0]?.name||'Click to select a file';});
}
$('sevenCoordinates').addEventListener('change',()=>{$('worldTransform').hidden=$('sevenCoordinates').value!=='custom';});
$('roomSelect').addEventListener('change',async()=>{
  const id=$('roomSelect').value;if(!id)return;
  try{
    state.statusEpoch++;await api('/api/rooms/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});
    resetRoom(id);await status();
  }catch(e){error(e.message);await loadRooms();}
});
async function uploadFile(file,kind,label,start,span){
  if(!file)throw new Error(`Select ${label}.`);
  const limit=kind==='model'?512*1024*1024:4*1024*1024*1024;
  if(file.size>limit)throw new Error(`${label} exceeds the size limit.`);
  const key=`room-upload:${kind}:${file.name}:${file.size}:${file.lastModified}`;
  let upload=null;
  try{const saved=localStorage.getItem(key);if(saved)upload=await api(`/api/uploads/${saved}`);}catch(e){}
  if(!upload){
    upload=await api('/api/uploads',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind,filename:file.name,size:file.size})});
    try{localStorage.setItem(key,upload.id);}catch(e){}
  }
  while(upload.offset<file.size){
    const end=Math.min(file.size,upload.offset+upload.chunk_bytes);let sent=false;
    for(let attempt=0;attempt<3&&!sent;attempt++){
      try{
        upload=await api(`/api/uploads/${upload.id}`,{method:'PUT',headers:{'Content-Type':'application/octet-stream','Upload-Offset':String(upload.offset)},body:file.slice(upload.offset,end)});
        sent=true;
      }catch(e){
        if(attempt===2)throw new Error(`${label} upload was interrupted. Keep the selected file and click again to resume.`);
        upload=await api(`/api/uploads/${upload.id}`);
        if(upload.offset>=end)sent=true;
      }
    }
    roomProgress(`${label}：${(upload.offset/1024/1024).toFixed(1)} / ${(file.size/1024/1024).toFixed(1)} MB`,start+span*upload.offset/file.size);
  }
  return upload.id;
}
$('roomForm').addEventListener('submit',async event=>{
  event.preventDefault();if(state.uploading||state.pendingRoom)return;
  roomError('');state.uploading=true;uploadControls();
  try{
    let G;
    if($('sevenCoordinates').value==='original')G=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
    if($('sevenCoordinates').value==='custom')G=JSON.parse($('worldTransform').value);
    const options={format:$('roomFormat').value,train_stride:Number($('trainStride').value),test_stride:Number($('testStride').value)};
    if(G)options.world_to_glb=G;
    const model=await uploadFile($('roomModelInput').files[0],'model','a 3D model',0,.25);
    const refs=await uploadFile($('roomRefsInput').files[0],'references','reference data',.25,.35);
    roomProgress('Upload complete. Checking files and preparing the room…',.6);
    const room=await api('/api/rooms',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('newRoomName').value,model_upload:model,references_upload:refs,options})});
    state.pendingRoom=room.id;try{localStorage.setItem('room-build',room.id);}catch(e){}
    state.statusEpoch++;resetRoom(room.id);await loadRooms();status();watchBuild(room.id);
  }catch(e){roomError(e.message);}
  finally{state.uploading=false;uploadControls();}
});
async function watchBuild(id){
  try{
    const room=await api(`/api/rooms/${id}`);
    if(room.reuse_id){
      if(state.currentRoom===id){state.statusEpoch++;await api('/api/rooms/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:room.reuse_id})});resetRoom(room.reuse_id);}
    }else if(room.phase==='error'){throw new Error(room.message);}
    else if(!room.ready){roomProgress(room.message||'Preparing room…',.6+.38*(room.progress||0));setTimeout(()=>watchBuild(id),1000);return;}
    state.pendingRoom=null;try{localStorage.removeItem('room-build');}catch(e){}
    roomProgress(room.reuse_id?'Existing reference library reused. Localization can begin.':'Room ready. Localization can begin.',1);
    $('roomDialog').close();uploadControls();await loadRooms();status();
  }catch(e){
    state.pendingRoom=null;try{localStorage.removeItem('room-build');}catch(e){}
    roomError(e.message);uploadControls();
  }
}
