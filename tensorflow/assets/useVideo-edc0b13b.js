import{B as d}from"./register_all_kernels-44fb38c1.js";import{m as l,f as u,h as g,g as m,q as v}from"./index-6dbbb0b1.js";function B({videoProceed:n,unmount:h=!0,videoRef:i,width:r=300,height:c=300}){const t=i?l(i)?i:u(i):u(document.createElement("video"));g(()=>{i||(t.value.width=r,t.value.height=c,t.value.autoplay="autoplay");const{offsetWidth:a,offsetHeight:s}=t.value;e.width=a,e.height=s});const e=m({status:!1,width:"",height:""});function f(){if(e.status)o();else if(d()){const a={video:!0,width:e.width,height:e.height};navigator.mediaDevices.getUserMedia(a).then(function(s){t.value.srcObject=s,t.value.addEventListener("loadeddata",function(){e.status=!0,n&&n()})})}}function o(){var a;(a=t.value.srcObject)==null||a.getTracks().forEach(s=>{s.stop()}),e.status=!1}return v(()=>{h&&o()}),{videoConfig:e,videoButtonClick:f,newVideoRef:t}}export{B as u};
