import{C as Fa,D as jr,E as X,F as so,G as io,H as Ea,I as oo,J as uo,K as Ca,L as On,M as Wr,N as mn,O as Na,P as kt,Q as Qt,R as Jt,S as Oa,U as Pa,V as Da,W as Ba,X as lo,Y as co,m as Be,Z as La,$ as rr,a0 as ja,a1 as $r,a2 as Wa,a3 as Ge,a4 as Ha,a5 as Gr,a6 as Va,a7 as mr,a8 as po,a9 as ho,aa as fo,ab as mo,ac as Ua,ad as za,ae as $a,af as go,ag as yo,ah as bo,ai as _o,aj as Ga,ak as vo,al as wo,am as gr,an as qa,ao as Ka,ap as Pn,aq as Xa,ar as ko,as as Io,at as Ya,au as Qa,av as Ja,aw as Za,ax as es,ay as ts,az as xo,aA as So,aB as Mo,aC as Ao,aD as To,aE as ns,aF as Ro,aG as Fo,aH as rs,aI as as,aJ as ss,aK as is,aL as Eo,aM as Co,aN as No,aO as Oo,aP as Po,aQ as Do,aR as os,aS as Bo,aT as Lo,aU as jo,aV as Wo,aW as Ho,aX as Vo,aY as Uo,aZ as us,a_ as zo,a$ as ls,b0 as cs,b1 as ps,b2 as $o,b3 as ds,b4 as Go,b5 as qo,b6 as hs,b7 as fs,b8 as ms,b9 as Ko,ba as gs,bb as Xo,bc as Yo,bd as Qo,be as ys,bf as Jo,bg as bs,bh as _s,bi as Zo,bj as eu,bk as tu,bl as nu,bm as ru,bn as vs,bo as ws,bp as ks,bq as Is,br as au,bs as su,bt as iu,bu as xs,bv as ou,bw as uu,bx as lu,by as cu,bz as Ss,bA as pu,bB as du,bC as hu,bD as fu,bE as mu,bF as Ms,bG as gu,bH as yu,bI as bu,bJ as _u,bK as vu,bL as qr,bM as wu,bN as ku,bO as Iu,bP as xu,bQ as Su,bR as Mu,bS as Au,bT as Tu,bU as Ru,bV as Fu,bW as As,bX as Ts,bY as Eu,bZ as Cu,b_ as Nu,b$ as Ou,c0 as Pu,c1 as Du,c2 as Rs,c3 as Bu,c4 as Lu,c5 as Fs,c6 as Es,c7 as Cs,c8 as ju,c9 as Wu,ca as Hu,cb as Jn,cc as zt,c as Tn,cd as Vu,ce as Uu,cf as Ns,cg as Kr,ch as Xr,ci as zu,l as fn,cj as Zn,T as gn,d as Hr,a as An,k as qt,t as Pe,h as Rn,ck as Rr,cl as $u,j as De,e as Fn,g as me,i as Kt,x as Rt,A as Vr,z as Gu,n as ze,f as Os,p as Me,q as Ve,cm as qu,v as hn,cn as Ku,co as Xu,r as Ue,u as Yr,w as Fr,y as Yu,cp as Qu,cq as Qn,cr as Ju,_ as Zu}from"./register_all_kernels-44fb38c1.js";import"./register_all_kernels-645548d3.js";import{u as el}from"./useVideo-edc0b13b.js";import{u as tl}from"./useCanvas-92fabe4f.js";import{d as nl,f as Yn,h as rl,c as Qr,i as $t,u as yr,t as al,j as sl,o as Jr,p as il,l as ol}from"./index-6dbbb0b1.js";function Ps(s,e){for(var a=0;a<e.length;a++){const o=e[a];if(typeof o!="string"&&!Array.isArray(o)){for(const i in o)if(i!=="default"&&!(i in s)){const t=Object.getOwnPropertyDescriptor(o,i);t&&Object.defineProperty(s,i,t.get?t:{enumerable:!0,get:()=>o[i]})}}}return Object.freeze(Object.defineProperty(s,Symbol.toStringTag,{value:"Module"}))}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var be;(function(s){s[s.float32=0]="float32",s[s.int32=1]="int32",s[s.bool=2]="bool",s[s.string=3]="string",s[s.complex64=4]="complex64"})(be||(be={}));var En;(function(s){s[s.linear=0]="linear",s[s.relu=1]="relu",s[s.relu6=2]="relu6",s[s.prelu=3]="prelu",s[s.leakyrelu=4]="leakyrelu",s[s.sigmoid=5]="sigmoid",s[s.elu=6]="elu"})(En||(En={}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Ds;function ul(s){Ds=s.wasm.cwrap(Fa,null,["number","array","number","number","array","number","number","number","number","number","number","number","number"])}function ll(s){const{inputs:e,backend:a,attrs:o}=s,{a:i,b:t,bias:u,preluActivationWeights:c}=e;if(i.dtype!=="float32"||t.dtype!=="float32")throw new Error("_FusedMatMul for non non-float32 tensors not yet supported.");const{transposeA:p,transposeB:h,activation:r,leakyreluAlpha:f}=o,b=a.dataIdMap.get(i.dataId).id,g=a.dataIdMap.get(t.dataId).id;let _=0;if(u!=null){const j=a.dataIdMap.get(u.dataId);if(j.shape.length!==1)throw new Error(`_FusedMatMul only supports rank-1 bias but got rank ${j.shape.length}.`);_=j.id}const v=c==null?0:a.dataIdMap.get(c.dataId).id,x=En[r];if(x==null)throw new Error(`${r} activation not yet supported for FusedConv2D in the wasm backend.`);const S=p?i.shape[2]:i.shape[1],k=h?t.shape[1]:t.shape[2],I=jr(i.shape.slice(0,-2),t.shape.slice(0,-2)),R=a.makeOutput([...I,S,k],i.dtype),N=a.dataIdMap.get(R.dataId).id,E=new Uint8Array(new Int32Array(i.shape).buffer),O=new Uint8Array(new Int32Array(t.shape).buffer);return Ds(b,E,i.shape.length,g,O,t.shape.length,p,h,x,_,v,f||0,N),R}const cl={kernelName:Fa,backendName:"wasm",setupFunc:ul,kernelFunc:ll};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function we(s,e){let a;function o(t){a=t.wasm.cwrap(s,null,["number","number","number"])}function i(t){const{backend:u,inputs:{x:c}}=t,p=u.dataIdMap.get(c.dataId).id,h=u.makeOutput(c.shape,e||c.dtype),r=u.dataIdMap.get(h.dataId).id;return X(h.shape)===0||a(p,be[c.dtype],r),h}return{kernelName:s,backendName:"wasm",setupFunc:o,kernelFunc:i}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pl=we(so);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ae(s,e,a){let o;function i(u){o=u.wasm.cwrap(s,null,["number","array","number","number","array","number","number","number"])}function t(u){const{backend:c,inputs:p}=u,{a:h,b:r}=p,f=c.dataIdMap.get(h.dataId).id,b=c.dataIdMap.get(r.dataId).id,g=a??h.dtype,_=jr(h.shape,r.shape),v=c.makeOutput(_,g);if(X(_)===0)return v;const x=new Uint8Array(new Int32Array(h.shape).buffer),S=new Uint8Array(new Int32Array(r.shape).buffer),k=c.dataIdMap.get(v.dataId).id;return(()=>o(f,x,h.shape.length,b,S,r.shape.length,be[h.dtype],k))(),v}return{kernelName:s,backendName:"wasm",setupFunc:i,kernelFunc:t}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dl=Ae(io);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Bs;function hl(s){Bs=s.wasm.cwrap(Ea,null,["array","number","number","number"])}function fl(s){const{inputs:e,backend:a}=s,o=a.makeOutput(e[0].shape,e[0].dtype);if(X(o.shape)===0)return o;const i=e.map(c=>a.dataIdMap.get(c.dataId).id),t=new Uint8Array(new Int32Array(i).buffer),u=a.dataIdMap.get(o.dataId).id;return Bs(t,i.length,be[o.dtype],u),o}const ml={kernelName:Ea,backendName:"wasm",setupFunc:hl,kernelFunc:fl};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ar(s){const{inputs:{x:e},backend:a}=s;if(e.dtype==="string")return uo(a.readSync(e.dataId),e.shape,e.dtype);const o=a.makeOutput(e.shape,e.dtype),i=a.typedArrayFromHeap(e);return a.typedArrayFromHeap(o).set(i),o}const gl={kernelName:oo,backendName:"wasm",kernelFunc:ar};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Ls;function yl(s){Ls=s.wasm.cwrap(Ca,null,["number","array","number","number","number","array","number"])}function Et(s){const{inputs:e,backend:a,attrs:o}=s,[i,t]=_l(e.x.shape,o.perm);let u=!0;for(let _=0;_<t.length;_++)t[_]!==_&&(u=!1);const c=bl(e.x.shape,o.perm),p={dataId:e.x.dataId,shape:i,dtype:e.x.dtype};if(u){const _=ar({inputs:e,backend:a});return _.shape=c,_}const h=a.makeOutput(c,p.dtype),r=a.dataIdMap.get(p.dataId).id,f=a.dataIdMap.get(h.dataId).id,b=new Uint8Array(new Int32Array(t).buffer),g=new Uint8Array(new Int32Array(p.shape).buffer);return Ls(r,g,p.shape.length,be[p.dtype],f,b,t.length),h}function bl(s,e){const a=new Array(s.length);for(let o=0;o<a.length;o++)a[o]=s[e[o]];return a}function _l(s,e){const a=[],o=[];for(let i=0;i<s.length;++i)s[i]!==1&&a.push(s[i]),s[e[i]]!==1&&o.push(e[i]);for(let i=0;i<o.length;++i){let t=-1;for(let u=0;u<o.length;++u)o[u]>=i&&(t===-1||o[t]>o[u])&&(t=u);o[t]=i}return[a,o]}const vl={kernelName:Ca,backendName:"wasm",kernelFunc:Et,setupFunc:yl};/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ct(s,e,a){const o=s.shape,i=s.shape.length,t=On(e,o);let u=t;const c=Wr(u,i);let p=null,h=!1;if(c!=null){const r=new Array(i);for(let g=0;g<r.length;g++)r[g]=o[c[g]];u=mn(u.length,i),p=Et({inputs:{x:s},attrs:{perm:c},backend:a});const f=a.dataIdMap.get(s.dataId).id;a.dataIdMap.get(p.dataId).id!==f&&(h=!0)}return{transposed:p,originalAxes:t,axes:u,inputWasTransposed:h}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let js;function wl(s){js=s.wasm.cwrap(Na,null,["number, number, number"])}function kl(s){const{backend:e,inputs:a,attrs:o}=s,{axis:i,keepDims:t}=o,{x:u}=a;let p=e.dataIdMap.get(u.dataId).id,h=u;const{transposed:r,axes:f,originalAxes:b,inputWasTransposed:g}=Ct(u,i,e);if(g){const I=e.dataIdMap.get(r.dataId).id;h=r,p=I}const _=h.shape.length;kt("all",f,_);const[v,x]=Qt(h.shape,f),S=X(x),k=e.makeOutput(v,u.dtype);if(X(h.shape)!==0){const I=e.dataIdMap.get(k.dataId).id;js(p,S,I)}if(g&&e.disposeData(r.dataId),t){const I=Jt(k.shape,b);k.shape=I}return k}const Il={kernelName:Na,backendName:"wasm",setupFunc:wl,kernelFunc:kl};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Ws;function xl(s){Ws=s.wasm.cwrap(Oa,null,["number, number, number"])}function Sl(s){const{backend:e,inputs:a,attrs:o}=s,{axis:i,keepDims:t}=o,{x:u}=a;let p=e.dataIdMap.get(u.dataId).id,h=u;const{transposed:r,axes:f,originalAxes:b,inputWasTransposed:g}=Ct(u,i,e);if(g){const I=e.dataIdMap.get(r.dataId).id;h=r,p=I}const _=h.shape.length;kt("any",f,_);const[v,x]=Qt(h.shape,f),S=X(x),k=e.makeOutput(v,u.dtype);if(X(h.shape)!==0){const I=e.dataIdMap.get(k.dataId).id;Ws(p,S,I)}if(g&&e.disposeData(r.dataId),t){const I=Jt(k.shape,b);k.shape=I}return k}const Ml={kernelName:Oa,backendName:"wasm",setupFunc:xl,kernelFunc:Sl};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Hs;function Al(s){Hs=s.wasm.cwrap(Pa,null,["number","number","number","number","number"])}function Tl(s){const{backend:e,inputs:a,attrs:o}=s,{axis:i}=o,{x:t}=a,u=e.dataIdMap.get(t.dataId).id;let c=u,p=t;const{transposed:h,axes:r,inputWasTransposed:f}=Ct(t,i,e);if(f){const S=e.dataIdMap.get(h.dataId).id;S!==u&&(p=h,c=S)}const b=p.shape.slice(0,-1),g=e.makeOutput(b,"int32"),_=e.dataIdMap.get(g.dataId).id,v=X(g.shape),x=p.shape[r[0]];return Hs(c,be[p.dtype],v,x,_),f&&e.disposeData(h.dataId),g}const Rl={kernelName:Pa,backendName:"wasm",kernelFunc:Tl,setupFunc:Al};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Vs;function Fl(s){Vs=s.wasm.cwrap(Da,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function El(s){const{inputs:e,attrs:a,backend:o}=s,i=e.x,t=o.dataIdMap.get(i.dataId).id,{filterSize:u,strides:c,pad:p,dimRoundingMode:h}=a,r=Ba(i.shape,u,c,1,p,h),f=r.filterHeight,b=r.filterWidth,g=r.padInfo.top,_=r.padInfo.right,v=r.padInfo.bottom,x=r.padInfo.left,S=r.strideHeight,k=r.strideWidth,I=r.inChannels;if(r.dataFormat!=="channelsLast")throw new Error(`wasm backend does not support dataFormat:'${r.dataFormat}'. Please use 'channelsLast'.`);if(r.dilationWidth!==1||r.dilationHeight!==1)throw new Error(`was backend only supports average pooling with dilation = [1, 1], got [${r.dilationHeight}, ${r.dilationWidth}].`);const R=o.makeOutput(r.outShape,"float32"),N=o.dataIdMap.get(R.dataId).id;return Vs(t,i.shape[0],i.shape[1],i.shape[2],f,b,g,_,v,x,S,k,I,N),R}const Cl={kernelName:Da,backendName:"wasm",setupFunc:Fl,kernelFunc:El};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Le(s){const{inputs:e,attrs:a}=s,{x:o}=e,{shape:i}=a,t=X(o.shape),u=co(i,t);return Be(t===X(u),()=>`new shape: ${u}, old shape: ${o.shape}. New shape and old shape must have the same number of elements.`),s.backend.incRef(o.dataId),{dataId:o.dataId,shape:u,dtype:o.dtype}}const Nl={kernelName:lo,backendName:"wasm",kernelFunc:Le};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Us;function Ol(s){Us=s.wasm.cwrap(La,null,["number","array","number","number","array","number","number","number","number"])}function Pl(s){const{inputs:e,backend:a,attrs:o}=s,{a:i,b:t}=e,{transposeA:u,transposeB:c}=o;if(i.dtype!=="float32"||t.dtype!=="float32")throw new Error("BatchMatMul for non non-float32 tensors not yet supported.");const p=i.shape.length,h=t.shape.length,r=u?i.shape[p-2]:i.shape[p-1],f=c?t.shape[h-1]:t.shape[h-2],b=u?i.shape[p-1]:i.shape[p-2],g=c?t.shape[h-2]:t.shape[h-1],_=i.shape.slice(0,-2),v=t.shape.slice(0,-2),x=X(_),S=X(v),I=jr(i.shape.slice(0,-2),t.shape.slice(0,-2)).concat([b,g]);Be(r===f,()=>`Error in matMul: inner shapes (${r}) and (${f}) of Tensors with shapes ${i.shape} and ${t.shape} and transposeA=${u} and transposeB=${c} must match.`);const R=u?[x,r,b]:[x,b,r],N=c?[S,g,f]:[S,f,g],E=Le({inputs:{x:i},backend:a,attrs:{shape:R}}),O=Le({inputs:{x:t},backend:a,attrs:{shape:N}}),j=a.dataIdMap.get(E.dataId).id,B=a.dataIdMap.get(O.dataId).id,D=u?E.shape[2]:E.shape[1],V=c?O.shape[1]:O.shape[2],q=Math.max(x,S),$=a.makeOutput([q,D,V],E.dtype),Y=a.dataIdMap.get($.dataId).id,Q=new Uint8Array(new Int32Array(E.shape).buffer),ee=new Uint8Array(new Int32Array(O.shape).buffer);return Us(j,Q,E.shape.length,B,ee,O.shape.length,u,c,Y),a.disposeData(E.dataId),a.disposeData(O.dataId),$.shape=I,$}const Dl={kernelName:La,backendName:"wasm",setupFunc:Ol,kernelFunc:Pl};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bl(s,e,a,o){const i=rr(a,X(e));if(o&&a!=="string"){let t=0;s.forEach(u=>{const c=X(u.shape);i.set(u.vals,t),t+=c})}else{let t=0;s.forEach(u=>{const c=a==="string"?ja(u.vals):u.vals;let p=0;for(let h=0;h<u.shape[0];++h){const r=h*e[1]+t;for(let f=0;f<u.shape[1];++f)i[r+f]=c[p++]}t+=u.shape[1]})}return i}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ll(s,e,a,o){const i=s===e,t=s<e&&a<0,u=e<s&&a>1;if(i||t||u)return $r(0,o);const c=Math.abs(Math.ceil((e-s)/a)),p=$r(c,o);e<s&&a===1&&(a=-1),p[0]=s;for(let h=1;h<p.length;h++)p[h]=p[h-1]+a;return p}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zr(s,e,a,o,i){const t=Wa(o,e,a),u=X(a),c=Ge(o);if(t){const f=Ha(e,c);return i==="string"?s.slice(f,f+u):s.subarray(f,f+u)}const p=i==="string"?ja(s):s,h=Gr(o,i,p),r=Gr(a,i);for(let f=0;f<r.size;++f){const b=r.indexToLoc(f),g=b.map((_,v)=>_+e[v]);r.set(h.get(...g),...b)}return i==="string"?Va(r.values):r.values}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class jl{constructor(e,a,o,i,t,u){this.separator=mr(e),this.nGramWidths=a,this.leftPad=mr(o),this.rightPad=mr(i),this.padWidth=t,this.preserveShort=u}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,a){const o=this.getPadWidth(a);return Math.max(0,e+2*o-a+1)}createNGrams(e,a,o,i,t,u){for(let c=0;c<t;++c){const p=this.getPadWidth(u),h=Math.max(0,p-c),r=Math.max(0,p-(t-(c+1))),f=u-(h+r),b=a+(h>0?0:c-p);let g=0;g+=h*this.leftPad.length;for(let k=0;k<f;++k)g+=e[b+k].length;g+=r*this.rightPad.length;const _=h+r+f-1;g+=_*this.separator.length,o[i+c]=new Uint8Array(g);const v=o[i+c];let x=0;const S=k=>k.forEach(I=>v[x++]=I);for(let k=0;k<h;++k)S(this.leftPad),S(this.separator);for(let k=0;k<f-1;++k)S(e[b+k]),S(this.separator);if(f>0){S(e[b+f-1]);for(let k=0;k<r;++k)S(this.separator),S(this.rightPad)}else{for(let k=0;k<r-1;++k)S(this.rightPad),S(this.separator);S(this.rightPad)}}}compute(e,a){const o=e.length,i=a.length;if(i>0){let p=a[0];if(p!==0)throw new Error(`First split value must be 0, got ${p}`);for(let h=1;h<i;++h){let r=a[h]>=p;if(r=r&&a[h]<=o,!r)throw new Error(`Invalid split value ${a[h]}, must be in [${p}, ${o}]`);p=a[h]}if(p!==o)throw new Error(`Last split value must be data size. Expected ${o}, got ${p}`)}const t=i-1,u=rr("int32",i);if(o===0||i===0){const p=new Array(o);for(let h=0;h<=t;++h)u[h]=0;return[p,u]}u[0]=0;for(let p=1;p<=t;++p){const h=a[p]-a[p-1];let r=0;this.nGramWidths.forEach(f=>{r+=this.getNumNGrams(h,f)}),this.preserveShort&&h>0&&r===0&&(r=1),u[p]=u[p-1]+r}const c=new Array(u[t]);for(let p=0;p<t;++p){const h=a[p];let r=u[p];if(this.nGramWidths.forEach(f=>{const b=a[p+1]-a[p],g=this.getNumNGrams(b,f);this.createNGrams(e,h,c,r,g,f),r+=g}),this.preserveShort&&r===u[p]){const f=a[p+1]-a[p];if(f===0)continue;const b=f+2*this.padWidth,g=1;this.createNGrams(e,h,c,r,g,b)}}return[c,u]}}function Wl(s,e,a,o,i,t,u,c){return new jl(a,o,i,t,u,c).compute(s,e)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hl(s,e,a,o){if(!s.length)return;if(e.length===0){for(let t=0;t<s.length;++t)o.push(s.subarray(t,t+1));return}if(e.length===1){const t=e[0];let u=s.indexOf(t);for(;u!==-1;){const c=s.subarray(0,u);(!a||c.length!==0)&&o.push(c),s=s.subarray(u+1),u=s.indexOf(t)}(!a||s.length!==0)&&o.push(s);return}let i=0;for(let t=0;t<s.length+1;t++)if(t===s.length||e.indexOf(s[t])!==-1){const u=s.subarray(i,t);(!a||u.length!==0)&&o.push(u),i=t+1}}function Vl(s,e,a){const o=s.length,i=[];let t=0,u=0;const c=new Array(o);for(let b=0;b<o;++b){const g=i.length;Hl(s[b],e,a,i);const _=i.length-g;c[b]=_,t+=_,u=Math.max(u,_)}const p=rr("int32",t*2),h=new Array(t),r=[o,u];let f=0;for(let b=0;b<o;++b)for(let g=0;g<c[b];++g)p[f*2]=b,p[f*2+1]=g,h[f]=i[f],++f;return[p,h,r]}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ul(s,e){const a=rr("int32",s.length);for(let o=0;o<s.length;++o)a[o]=po(s[o]).modulo(e).getLowBitsUnsigned();return a}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yt(s){const{inputs:{x:e},attrs:{begin:a,size:o},backend:i}=s,[t,u]=fo(e,a,o),c=Wa(e.shape,t,u),p=i.readSync(e.dataId),h=i.makeOutput(u,e.dtype),r=Ge(e.shape),f=i.dataIdMap.get(h.dataId);if(c){const _=Ha(t,r);return e.dtype==="string"?f.stringBytes=p.slice(_,_+X(u)):i.typedArrayFromHeap(h).set(p.subarray(_,_+X(u))),h}if(e.dtype==="string"){const _=Zr(p,t,u,e.shape,e.dtype);return f.stringBytes=_,h}const b=i.typedArrayFromHeap(h),g=e.shape.length;if(g===2)zl(p,r[0],b,t,u);else if(g===3)$l(p,r[0],r[1],b,t,u);else if(g===4)Gl(p,r[0],r[1],r[2],b,t,u);else{const _=Zr(p,t,u,e.shape,e.dtype);b.set(_)}return h}function zl(s,e,a,o,i){let t=0;const u=o[0],c=o[1],p=u+i[0];for(let h=u;h<p;h++){const r=h*e+c;a.set(s.subarray(r,r+i[1]),t),t+=i[1]}}function $l(s,e,a,o,i,t){let u=0;const c=i[0],p=i[1],h=i[2],r=c+t[0],f=p+t[1];for(let b=c;b<r;b++)for(let g=p;g<f;g++){const _=b*e+g*a+h;o.set(s.subarray(_,_+t[2]),u),u+=t[2]}}function Gl(s,e,a,o,i,t,u){let c=0;const p=t[0],h=t[1],r=t[2],f=p+u[0],b=h+u[1],g=r+u[2],_=t[3];for(let v=p;v<f;v++)for(let x=h;x<b;x++)for(let S=r;S<g;S++){const k=v*e+x*a+S*o+_;i.set(s.subarray(k,k+u[3]),c),c+=u[3]}}const ql={kernelName:ho,backendName:"wasm",kernelFunc:Yt};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kl(s){const{inputs:e,backend:a,attrs:o}=s,{x:i}=e,{blockShape:t,crops:u}=o,c=t.reduce((S,k)=>S*k),p=Ua(i.shape,t,c),h=za(p.length,t.length),r=$a(i.shape,t,c),f=go(u,t.length),b=yo(r,u,t.length),g=Le({inputs:{x:i},backend:a,attrs:{shape:p}}),_=Et({inputs:{x:g},backend:a,attrs:{perm:h}}),v=Le({inputs:{x:_},backend:a,attrs:{shape:r}}),x=Yt({inputs:{x:v},backend:a,attrs:{begin:f,size:b}});return a.disposeData(g.dataId),a.disposeData(_.dataId),a.disposeData(g.dataId),x}const Xl={kernelName:mo,backendName:"wasm",kernelFunc:Kl};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yn(s){const{inputs:{x:e},attrs:{dtype:a},backend:o}=s,i=o.makeOutput(e.shape,a),t=o.typedArrayFromHeap(e);return o.typedArrayFromHeap(i).set(t),i}const Yl={kernelName:bo,backendName:"wasm",kernelFunc:yn};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ql=we(_o);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let zs;function Jl(s){zs=s.wasm.cwrap(Ga,null,["number","number","number","number"])}function Zl(s){const{inputs:e,backend:a,attrs:o}=s,{x:i}=e,{clipValueMin:t,clipValueMax:u}=o,c=a.dataIdMap.get(i.dataId).id,p=a.makeOutput(i.shape,i.dtype),h=a.dataIdMap.get(p.dataId).id;return zs(c,t,u,h),p}const ec={kernelName:Ga,backendName:"wasm",setupFunc:Jl,kernelFunc:Zl};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $s(s){const{inputs:e,backend:a}=s,o=On(s.attrs.axis,e[0].shape)[0],i=e.map(g=>g.shape);wo(i,o);let t=gr(e.map(g=>g.shape),o);const u=e.filter(g=>X(g.shape)>0);if(u.length===1)return ar({inputs:{x:u[0]},backend:a});const c=a.makeOutput(t,e[0].dtype);if(X(t)===0)return c;if(u[0].dtype==="string"){const g=u.map(I=>{const N=[-1,X(I.shape.slice(o))];return Le({inputs:{x:I},backend:a,attrs:{shape:N}})}),_=g.map(I=>({vals:a.readSync(I.dataId),shape:I.shape}));t=gr(g.map(I=>I.shape),1);const v=g[0].shape[0]===1,x=Bl(_,t,e[0].dtype,v),S=gr(u.map(I=>I.shape),o);c.shape=S;const k=a.dataIdMap.get(c.dataId);return k.stringBytes=Va(x),g.forEach(I=>a.disposeData(I.dataId)),c}const p=X(u[0].shape.slice(0,o));let h=0;const r=u.map(g=>{const _=X(g.shape.slice(o));return h+=_,_}),f=u.map(g=>a.typedArrayFromHeap(g)),b=a.typedArrayFromHeap(c);for(let g=0;g<p;g++){let _=g*h;for(let v=0;v<f.length;v++){const x=r[v],S=g*x,k=f[v].subarray(S,S+x);b.set(k,_),_+=x}}return c}const tc={kernelName:vo,backendName:"wasm",kernelFunc:$s};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Gs;function nc(s){Gs=s.wasm.cwrap(qa,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function rc(s){const{inputs:e,attrs:a,backend:o}=s,{x:i,filter:t}=e,u=o.dataIdMap.get(i.dataId).id,c=o.dataIdMap.get(t.dataId).id,{strides:p,dilations:h,pad:r,dimRoundingMode:f,dataFormat:b}=a,g=Ka(b),_=Pn(i.shape,t.shape,p,h,r,f,!1,g),v=_.filterHeight,x=_.filterWidth,S=_.padInfo.top,k=_.padInfo.right,I=_.padInfo.bottom,R=_.padInfo.left,N=_.dilationHeight,E=_.dilationWidth,O=_.strideHeight,j=_.strideWidth,B=_.inChannels,D=_.outChannels,V=_.padInfo.type==="SAME"?1:0;if(_.dataFormat!=="channelsLast")throw new Error(`wasm backend Conv2D does not support dataFormat:'${_.dataFormat}'. Please use 'channelsLast'.`);const q=o.makeOutput(_.outShape,"float32"),$=o.dataIdMap.get(q.dataId).id;return Gs(u,i.shape[0],i.shape[1],i.shape[2],c,v,x,S,k,I,R,V,N,E,O,j,B,D,$),q}const ac={kernelName:qa,backendName:"wasm",setupFunc:nc,kernelFunc:rc};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let qs;function sc(s){qs=s.wasm.cwrap(Xa,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function ic(s){const{backend:e,inputs:a,attrs:o}=s,{dy:i,filter:t}=a,{strides:u,pad:c,dataFormat:p,dimRoundingMode:h,inputShape:r}=o,f=1,b=Ka(p),g=Pn(r,t.shape,u,f,c,h,!1,b),{batchSize:_,filterHeight:v,filterWidth:x,inChannels:S,inHeight:k,inWidth:I,outChannels:R,outHeight:N,outWidth:E,strideHeight:O,strideWidth:j}=g,B=v-1-g.padInfo.top,D=x-1-g.padInfo.left,V=g.dataFormat==="channelsLast",q=Ge(g.inShape),$=Ge(i.shape),[Y,Q,ee]=Ge(t.shape),ce=q[0],ae=V?q[1]:q[2],xe=V?q[2]:1,Te=V?1:q[1],Ee=$[0],qe=V?$[1]:$[2],ke=V?$[2]:1,yt=V?1:$[1],bt=e.makeOutput(g.inShape,"float32"),nt=e.dataIdMap.get(bt.dataId).id,It=e.dataIdMap.get(i.dataId).id,ge=e.dataIdMap.get(t.dataId).id;return qs(It,ge,_,v,x,k,I,S,N,E,R,O,j,B,D,Y,Q,ee,ce,ae,xe,Te,Ee,qe,ke,yt,nt),bt}const oc={kernelName:Xa,backendName:"wasm",setupFunc:sc,kernelFunc:ic};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uc=we(ko);/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lc=we(Io);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Er;(function(s){s[s.bilinear=0]="bilinear",s[s.nearest=1]="nearest"})(Er||(Er={}));let Ks;function cc(s){Ks=s.wasm.cwrap(Ya,null,["number","number","number","number","array","number","number","number","number","number"])}function pc(s){const{backend:e,inputs:a,attrs:o}=s,{method:i,extrapolationValue:t,cropSize:u}=o,{image:c,boxes:p,boxInd:h}=a,r=p.shape[0],[f,b]=u,g=[r,f,b,c.shape[3]];let _=e.dataIdMap.get(c.dataId),v;c.dtype!=="float32"&&(v=yn({backend:e,inputs:{x:c},attrs:{dtype:"float32"}}),_=e.dataIdMap.get(v.dataId));const x=_.id,S=e.dataIdMap.get(p.dataId).id,k=e.dataIdMap.get(h.dataId).id,I=e.makeOutput(g,"float32"),R=e.dataIdMap.get(I.dataId).id,N=new Uint8Array(new Int32Array(c.shape).buffer);return Ks(x,S,k,r,N,f,b,Er[i],t,R),v!=null&&e.disposeData(v.dataId),I}const dc={kernelName:Ya,backendName:"wasm",setupFunc:cc,kernelFunc:pc};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Xs;function hc(s){Xs=s.wasm.cwrap(Qa,null,["number","number","number","number","number","number"])}function fc(s){const{inputs:e,backend:a,attrs:o}=s,{x:i}=e,{axis:t,exclusive:u,reverse:c}=o,p=i.shape.length;Be(i.dtype==="float32"||i.dtype==="int32",()=>`cumprod does not support ${i.dtype} tensors in the WASM backend`);const h=Wr([t],p);let r=i;h!==null&&(r=Et({inputs:{x:i},attrs:{perm:h},backend:a}));const f=mn(1,p)[0];kt("cumprod",[f],p);const b=a.makeOutput(r.shape,r.dtype),g=r.shape[f],_=a.dataIdMap.get(r.dataId).id,v=a.dataIdMap.get(b.dataId).id;Xs(_,u?1:0,c?1:0,g,v,be[i.dtype]);let x=b;if(h!==null){const S=Ja(h);x=Et({inputs:{x:b},attrs:{perm:S},backend:a}),a.disposeData(r.dataId),a.disposeData(b.dataId)}return x}const mc={kernelName:Qa,backendName:"wasm",setupFunc:hc,kernelFunc:fc};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Ys;function gc(s){Ys=s.wasm.cwrap(Za,null,["number","number","number","number","number","number"])}function yc(s){const{inputs:e,backend:a,attrs:o}=s,{x:i}=e,{axis:t,exclusive:u,reverse:c}=o,p=i.shape.length;Be(i.dtype==="float32"||i.dtype==="int32",()=>`cumsum does not support ${i.dtype} tensors in the WASM backend`);const h=Wr([t],p);let r=i;h!==null&&(r=Et({inputs:{x:i},attrs:{perm:h},backend:a}));const f=mn(1,p)[0];kt("cumsum",[f],p);const b=a.makeOutput(r.shape,r.dtype),g=r.shape[f],_=a.dataIdMap.get(r.dataId).id,v=a.dataIdMap.get(b.dataId).id;Ys(_,u?1:0,c?1:0,g,v,be[i.dtype]);let x=b;if(h!==null){const S=Ja(h);x=Et({inputs:{x:b},attrs:{perm:S},backend:a}),a.disposeData(r.dataId),a.disposeData(b.dataId)}return x}const bc={kernelName:Za,backendName:"wasm",setupFunc:gc,kernelFunc:yc};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Qs;function _c(s){Qs=s.wasm.cwrap(es,null,["number","number","number","array","number","array","array","number","number"])}function vc(s){const{backend:e,inputs:a,attrs:o}=s,{x:i}=a,{blockSize:t,dataFormat:u}=o,c=i.shape[0],p=u==="NHWC"?i.shape[1]:i.shape[2],h=u==="NHWC"?i.shape[2]:i.shape[3],r=u==="NHWC"?i.shape[3]:i.shape[1],f=p*t,b=h*t,g=r/(t*t),_=u==="NHWC"?[c,f,b,g]:[c,g,f,b],v=e.makeOutput(_,"float32"),S=e.dataIdMap.get(i.dataId).id,k=new Uint8Array(new Int32Array(Ge(i.shape)).buffer),I=new Uint8Array(new Int32Array(_).buffer),R=new Uint8Array(new Int32Array(Ge(_)).buffer),N=e.dataIdMap.get(v.dataId).id;return Qs(S,t,u==="NHWC"?1:0,k,i.shape.length-1,I,R,_.length,N),v}const wc={kernelName:es,backendName:"wasm",setupFunc:_c,kernelFunc:vc};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Js;function kc(s){Js=s.wasm.cwrap(ts,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function Ic(s){const{inputs:e,attrs:a,backend:o}=s,{x:i,filter:t}=e,u=o.dataIdMap.get(i.dataId).id,c=o.dataIdMap.get(t.dataId).id,{strides:p,dilations:h,pad:r,dimRoundingMode:f}=a,b=h??[1,1],g=Pn(i.shape,t.shape,p,b,r,f,!0),_=g.filterHeight,v=g.filterWidth,x=g.padInfo.top,S=g.padInfo.right,k=g.padInfo.bottom,I=g.padInfo.left,R=g.dilationHeight,N=g.dilationWidth,E=g.strideHeight,O=g.strideWidth,j=g.inChannels,B=g.outChannels,D=g.padInfo.type==="SAME"?1:0;if(g.dataFormat!=="channelsLast")throw new Error(`wasm backend DepthwiseConv2dNative does not support dataFormat:'${g.dataFormat}'. Please use 'channelsLast'.`);const V=o.makeOutput(g.outShape,"float32"),q=o.dataIdMap.get(V.dataId).id;return Js(u,i.shape[0],i.shape[1],i.shape[2],c,_,v,x,S,k,I,D,R,N,E,O,j,B,q),V}const xc={kernelName:ts,backendName:"wasm",setupFunc:kc,kernelFunc:Ic};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sc=we(xo);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mc=!1,Ac=Ae(So,Mc,"bool");/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tc=we(Mo,"float32");/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cr(s){const{inputs:e,attrs:a,backend:o}=s,{input:i}=e,{dim:t}=a,u=i.shape.length,c=i.shape.slice();let p=t;return t<0&&(Be(-(u+1)<=t,()=>`Axis must be in the interval [${-(u+1)}, ${u}]`),p=u+t+1),c.splice(p,0,1),Le({inputs:{x:i},backend:o,attrs:{shape:c}})}const Rc={kernelName:Ao,backendName:"wasm",kernelFunc:Cr};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zs(s){const{attrs:{shape:e,value:a,dtype:o},backend:i}=s,t=i.makeOutput(e,o);return i.typedArrayFromHeap(t).fill(a),t}const Fc={kernelName:To,backendName:"wasm",kernelFunc:Zs};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let ei;function Ec(s){ei=s.wasm.cwrap(ns,null,["number","number","number","number","number","number"])}function Cc(s){const{inputs:e,backend:a}=s,{image:o}=e,i=a.makeOutput(o.shape,o.dtype),t=a.dataIdMap.get(o.dataId).id,u=a.dataIdMap.get(i.dataId).id,[c,p,h,r]=o.shape;return ei(t,c,p,h,r,u),i}const Nc={kernelName:ns,backendName:"wasm",kernelFunc:Cc,setupFunc:Ec};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Oc=we(Ro);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pc=Ae(Fo);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let ti;function Dc(s){ti=s.wasm.cwrap(rs,null,["number","number","number","number","number","number","number"])}function Bc(s){const{backend:e,inputs:a,attrs:o}=s,{varianceEpsilon:i}=o,{x:t,mean:u,variance:c,offset:p,scale:h}=a,r=e.dataIdMap.get(t.dataId).id,f=e.dataIdMap.get(u.dataId).id,b=e.dataIdMap.get(c.dataId).id,g=p!=null?e.dataIdMap.get(p.dataId).id:0,_=h!=null?e.dataIdMap.get(h.dataId).id:0,v=e.makeOutput(t.shape,t.dtype);if(X(t.shape)===0)return v;const x=e.dataIdMap.get(v.dataId).id;return ti(r,f,b,g,_,i,x),v}const Lc={kernelName:rs,backendName:"wasm",setupFunc:Dc,kernelFunc:Bc};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let ni;function jc(s){ni=s.wasm.cwrap(as,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function Wc(s){const{inputs:e,attrs:a,backend:o}=s,{x:i,filter:t,bias:u,preluActivationWeights:c}=e,{strides:p,pad:h,dilations:r,dataFormat:f,dimRoundingMode:b,activation:g,leakyreluAlpha:_}=a,v=Pn(i.shape,t.shape,p,r,h,b),x=En[g];if(x==null)throw new Error(`${g} activation not yet supported for FusedConv2D in the wasm backend.`);const S=o.dataIdMap.get(i.dataId).id,k=o.dataIdMap.get(t.dataId).id,I=v.outChannels;let R=0;if(u!=null){const ke=o.dataIdMap.get(u.dataId);if(ke.shape.length!==1)throw new Error(`FusedConv2D only supports rank-1 bias but got rank ${ke.shape.length}.`);if(ke.shape[0]!==I)throw new Error(`FusedConv2D bias shape (${ke.shape}) does not match the number of output channels (${I})`);R=ke.id}const N=v.filterHeight,E=v.filterWidth,O=v.padInfo.top,j=v.padInfo.right,B=v.padInfo.bottom,D=v.padInfo.left,V=v.dilationHeight,q=v.dilationWidth,$=v.strideHeight,Y=v.strideWidth,Q=v.inChannels,ee=v.padInfo.type==="SAME"?1:0,ce=v.batchSize,ae=v.inHeight,xe=v.inWidth;if(f!=="NHWC")throw new Error(`wasm backend FusedConv2D does not support dataFormat:'${f}'. Please use 'NHWC'.`);const Te=o.makeOutput(v.outShape,"float32"),Ee=o.dataIdMap.get(Te.dataId).id,qe=c==null?0:o.dataIdMap.get(c.dataId).id;return ni(S,ce,ae,xe,k,N,E,R,O,j,B,D,ee,V,q,$,Y,Q,I,x,qe,_||0,Ee),Te}const Hc={kernelName:as,backendName:"wasm",setupFunc:jc,kernelFunc:Wc};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let ri;function Vc(s){ri=s.wasm.cwrap(ss,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function Uc(s){const{inputs:e,attrs:a,backend:o}=s,{x:i,filter:t,bias:u,preluActivationWeights:c}=e,{strides:p,pad:h,dilations:r,dataFormat:f,dimRoundingMode:b,activation:g,leakyreluAlpha:_}=a,v=Pn(i.shape,t.shape,p,r,h,b,!0),x=En[g];if(x==null)throw new Error(`${g} activation not yet supported for FusedDepthwiseConv2D in the wasm backend.`);const S=o.dataIdMap.get(i.dataId).id,k=o.dataIdMap.get(t.dataId).id,I=v.outChannels;let R=0;if(u!=null){const ke=o.dataIdMap.get(u.dataId);if(ke.shape.length!==1)throw new Error(`FusedDepthwiseConv2D only supports rank-1 bias but got rank ${ke.shape.length}.`);if(ke.shape[0]!==I)throw new Error(`FusedDepthwiseConv2D bias shape (${ke.shape}) does not match the number of output channels (${I})`);R=ke.id}const N=v.filterHeight,E=v.filterWidth,O=v.padInfo.top,j=v.padInfo.right,B=v.padInfo.bottom,D=v.padInfo.left,V=v.dilationHeight,q=v.dilationWidth,$=v.strideHeight,Y=v.strideWidth,Q=v.inChannels,ee=v.padInfo.type==="SAME"?1:0,ce=v.batchSize,ae=v.inHeight,xe=v.inWidth;if(f!=="NHWC")throw new Error(`wasm backend FusedDepthwiseConv2D does not support dataFormat:'${f}'. Please use 'NHWC'.`);const Te=o.makeOutput(v.outShape,"float32"),Ee=o.dataIdMap.get(Te.dataId).id,qe=c==null?0:o.dataIdMap.get(c.dataId).id;return ri(S,ce,ae,xe,k,N,E,R,O,j,B,D,ee,V,q,$,Y,Q,I,x,qe,_||0,Ee),Te}const zc={kernelName:ss,backendName:"wasm",setupFunc:Vc,kernelFunc:Uc};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let ai;function $c(s){ai=s.wasm.cwrap(is,null,["number","number","number","number","number","number","array","number"])}function Gc(s){const{backend:e,inputs:a}=s,{params:o,indices:i}=a,[t,u,c,p]=Eo(o,i),h=e.makeOutput(t,o.dtype);if(u===0)return h;const r=i.shape,f=r[r.length-1],g=e.dataIdMap.get(o.dataId).id,v=e.dataIdMap.get(i.dataId).id,x=new Uint8Array(new Int32Array(p).buffer),S=e.dataIdMap.get(h.dataId).id;return ai(g,be[o.dtype],v,u,f,c,x,S),h}const qc={kernelName:is,backendName:"wasm",setupFunc:$c,kernelFunc:Gc};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let si;function Kc(s){si=s.wasm.cwrap("Gather",null,["number","number","array","number","number","number","array","number"])}function Xc(s){const{backend:e,inputs:a,attrs:o}=s,{x:i,indices:t}=a,{axis:u,batchDims:c}=o,p=On(u,i.shape)[0],h=e.readSync(t.dataId),r=i.shape[p];for(let B=0;B<h.length;++B){const D=h[B];Be(D<=r-1&&D>=0,()=>`GatherV2: the index value ${D} is not in [0, ${r-1}]`)}const f=No(i,t,p,c),b=Le({inputs:{x:i},attrs:{shape:[f.batchSize,f.outerSize,f.dimSize,f.sliceSize]},backend:e}),g=X(t.shape),_=Le({inputs:{x:t},attrs:{shape:[f.batchSize,g/f.batchSize]},backend:e}),v=[f.batchSize,f.outerSize,g/f.batchSize,f.sliceSize],x=e.makeOutput(v,i.dtype);if(X(i.shape)===0)return x;const S=b.shape.length-1,I=e.dataIdMap.get(b.dataId).id,N=e.dataIdMap.get(_.dataId).id,E=e.dataIdMap.get(x.dataId).id,O=new Uint8Array(new Int32Array(Ge(b.shape)).buffer),j=new Uint8Array(new Int32Array(Ge(v)).buffer);return si(I,be[i.dtype],O,S,N,f.batchSize,j,E),e.disposeData(b.dataId),e.disposeData(_.dataId),x.shape=f.outputShape,x}const Yc={kernelName:Co,backendName:"wasm",setupFunc:Kc,kernelFunc:Xc};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qc=!1,Jc=Ae(Oo,Qc,"bool");/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zc=!1,ep=Ae(Po,Zc,"bool");/**
 * @license
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tp=we(Do,"bool");/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let ii;function np(s){ii=s.wasm.cwrap(os,null,["number","number","number","number"])}function rp(s){const{inputs:{x:e},attrs:{alpha:a},backend:o}=s,i=o.dataIdMap.get(e.dataId).id,t=o.makeOutput(e.shape,"float32");if(X(e.shape)!==0){const u=o.dataIdMap.get(t.dataId).id;ii(i,be[e.dtype],a,u)}return t}const ap={kernelName:os,backendName:"wasm",setupFunc:np,kernelFunc:rp};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sp=!1,ip=Ae(Bo,sp,"bool");/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const op=!1,up=Ae(Lo,op,"bool");/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lp=we(jo);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cp=!1,pp=Ae(Wo,cp,"bool");/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dp=we(Ho);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hp=!1,fp=Ae(Vo,hp,"bool");/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mp=!1,gp=Ae(Uo,mp,"bool");/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let oi;function yp(s){oi=s.wasm.cwrap(us,null,["number","number","number","number"])}function bp(s){const{backend:e,inputs:a,attrs:o}=s,{reductionIndices:i,keepDims:t}=o,{x:u}=a;let p=e.dataIdMap.get(u.dataId).id,h=u;const{transposed:r,axes:f,originalAxes:b,inputWasTransposed:g}=Ct(u,i,e);if(g){const I=e.dataIdMap.get(r.dataId).id;h=r,p=I}const _=h.shape.length;kt("max",f,_);const[v,x]=Qt(h.shape,f),S=X(x),k=e.makeOutput(v,u.dtype);if(X(h.shape)!==0){const I=e.dataIdMap.get(k.dataId).id;oi(p,be[u.dtype],S,I)}if(g&&e.disposeData(r.dataId),t){const I=Jt(k.shape,b);k.shape=I}return k}const _p={kernelName:us,backendName:"wasm",setupFunc:yp,kernelFunc:bp};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vp=Ae(zo);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let ui;function wp(s){ui=s.wasm.cwrap(ls,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function kp(s){const{inputs:e,attrs:a,backend:o}=s,i=e.x,t=o.dataIdMap.get(i.dataId).id;Be(i.dtype==="float32",()=>`Error in MaxPool: only float32 input is supported. Got ${i.dtype}.`);const{filterSize:u,strides:c,pad:p,dimRoundingMode:h}=a,r=Ba(i.shape,u,c,1,p,h),f=r.filterHeight,b=r.filterWidth,g=r.padInfo.top,_=r.padInfo.right,v=r.padInfo.bottom,x=r.padInfo.left,S=r.dilationHeight,k=r.dilationWidth,I=r.strideHeight,R=r.strideWidth,N=r.inChannels,E=r.outChannels;if(r.dataFormat!=="channelsLast")throw new Error(`wasm backend does not support dataFormat:'${r.dataFormat}'. Please use 'channelsLast'.`);const O=o.makeOutput(r.outShape,"float32"),j=o.dataIdMap.get(O.dataId).id;return ui(t,i.shape[0],i.shape[1],i.shape[2],f,b,g,_,v,x,S,k,I,R,N,E,j),O}const Ip={kernelName:ls,backendName:"wasm",setupFunc:wp,kernelFunc:kp};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let li;function xp(s){li=s.wasm.cwrap(cs,null,["number, number, number"])}function Sp(s){const{backend:e,inputs:a,attrs:o}=s,{axis:i,keepDims:t}=o,{x:u}=a,c=e.dataIdMap.get(u.dataId).id;let p=c,h=u;const{transposed:r,axes:f,originalAxes:b,inputWasTransposed:g}=Ct(u,i,e);let _=f;if(g){const R=e.dataIdMap.get(r.dataId).id;R!==c&&(h=r,p=R,_=mn(_.length,h.shape.length))}kt("mean",_,h.shape.length);const[v,x]=Qt(h.shape,_),S=X(x);let k=h;h.dtype!=="float32"&&(k=yn({backend:e,inputs:{x:h},attrs:{dtype:"float32"}}),p=e.dataIdMap.get(k.dataId).id);const I=e.makeOutput(v,"float32");if(X(h.shape)!==0){const R=e.dataIdMap.get(I.dataId).id;li(p,S,R)}if(g&&e.disposeData(r.dataId),t){const R=Jt(I.shape,b);I.shape=R}return h.dtype!=="float32"&&e.disposeData(k.dataId),I}const Mp={kernelName:cs,backendName:"wasm",setupFunc:xp,kernelFunc:Sp};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let ci;function Ap(s){ci=s.wasm.cwrap(ps,null,["number","number","number","number"])}function Tp(s){const{backend:e,inputs:a,attrs:o}=s,{axis:i,keepDims:t}=o,{x:u}=a,c=e.dataIdMap.get(u.dataId).id;let p=c,h=u;const{transposed:r,axes:f,originalAxes:b,inputWasTransposed:g}=Ct(u,i,e);if(g){const I=e.dataIdMap.get(r.dataId).id;I!==c&&(h=r,p=I)}const _=h.shape.length;kt("min",f,_);const[v,x]=Qt(h.shape,f),S=X(x),k=e.makeOutput(v,h.dtype);if(X(h.shape)!==0){const I=e.dataIdMap.get(k.dataId).id;ci(p,be[u.dtype],S,I)}if(g&&e.disposeData(r.dataId),t){const I=Jt(k.shape,b);k.shape=I}return k}const Rp={kernelName:ps,backendName:"wasm",setupFunc:Ap,kernelFunc:Tp};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fp=Ae($o);/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Nr;(function(s){s[s.reflect=0]="reflect",s[s.symmetric=1]="symmetric"})(Nr||(Nr={}));let pi;function Ep(s){pi=s.wasm.cwrap(ds,null,["number","array","number","number","array","array","number","number"])}function Cp(s){const{inputs:{x:e},backend:a,attrs:{paddings:o,mode:i}}=s,t=o.map((_,v)=>_[0]+e.shape[v]+_[1]),u=a.dataIdMap.get(e.dataId).id,c=a.makeOutput(t,e.dtype),p=a.dataIdMap.get(c.dataId).id,h=new Uint8Array(new Int32Array(e.shape).buffer),r=o.map(_=>_[0]),f=o.map(_=>_[1]),b=new Uint8Array(new Int32Array(r).buffer),g=new Uint8Array(new Int32Array(f).buffer);return pi(u,h,e.shape.length,be[e.dtype],b,g,Nr[i],p),c}const Np={kernelName:ds,backendName:"wasm",kernelFunc:Cp,setupFunc:Ep};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Op=Ae(Go);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pp=we(qo);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ur(s,e){const a=new Int32Array(s.wasm.HEAPU8.buffer,e,4),o=a[0],i=a[1],t=a[2],u=a[3];return s.wasm._free(e),{pSelectedIndices:o,selectedSize:i,pSelectedScores:t,pValidOutputs:u}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let di;function Dp(s){di=s.wasm.cwrap(hs,"number",["number","number","number","number","number"])}function Bp(s){const{backend:e,inputs:a,attrs:o}=s,{iouThreshold:i,maxOutputSize:t,scoreThreshold:u}=o,{boxes:c,scores:p}=a,h=e.dataIdMap.get(c.dataId).id,r=e.dataIdMap.get(p.dataId).id,f=di(h,r,t,i,u),{pSelectedIndices:b,selectedSize:g,pSelectedScores:_,pValidOutputs:v}=Ur(e,f);return e.wasm._free(_),e.wasm._free(v),e.makeOutput([g],"int32",b)}const Lp={kernelName:hs,backendName:"wasm",setupFunc:Dp,kernelFunc:Bp};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let hi;function jp(s){hi=s.wasm.cwrap(fs,"number",["number","number","number","number","number","bool"])}function Wp(s){const{backend:e,inputs:a,attrs:o}=s,{iouThreshold:i,maxOutputSize:t,scoreThreshold:u,padToMaxOutputSize:c}=o,{boxes:p,scores:h}=a,r=e.dataIdMap.get(p.dataId).id,f=e.dataIdMap.get(h.dataId).id,b=hi(r,f,t,i,u,c),{pSelectedIndices:g,selectedSize:_,pSelectedScores:v,pValidOutputs:x}=Ur(e,b);e.wasm._free(v);const S=e.makeOutput([_],"int32",g),k=e.makeOutput([],"int32",x);return[S,k]}const Hp={kernelName:fs,backendName:"wasm",setupFunc:jp,kernelFunc:Wp};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let fi;function Vp(s){fi=s.wasm.cwrap(ms,"number",["number","number","number","number","number","number"])}function Up(s){const{backend:e,inputs:a,attrs:o}=s,{iouThreshold:i,maxOutputSize:t,scoreThreshold:u,softNmsSigma:c}=o,{boxes:p,scores:h}=a,r=e.dataIdMap.get(p.dataId).id,f=e.dataIdMap.get(h.dataId).id,b=fi(r,f,t,i,u,c),{pSelectedIndices:g,selectedSize:_,pSelectedScores:v,pValidOutputs:x}=Ur(e,b);e.wasm._free(x);const S=e.makeOutput([_],"int32",g),k=e.makeOutput([_],"float32",v);return[S,k]}const zp={kernelName:ms,backendName:"wasm",setupFunc:Vp,kernelFunc:Up};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $p=!1,Gp=Ae(Ko,$p,"bool");/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let mi;function qp(s){mi=s.wasm.cwrap(gs,null,["number","number","number","number","number"])}function Kp(s){const{inputs:e,backend:a,attrs:o}=s,{indices:i}=e,{dtype:t,depth:u,onValue:c,offValue:p}=o,h=a.makeOutput([...i.shape,u],t),r=a.dataIdMap.get(h.dataId).id,b=a.dataIdMap.get(i.dataId).id;return mi(b,u,c,p,r),h}const Xp={kernelName:gs,backendName:"wasm",setupFunc:qp,kernelFunc:Kp};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yp(s){const{inputs:{x:e},backend:a}=s,o=a.makeOutput(e.shape,e.dtype);return a.typedArrayFromHeap(o).fill(1),o}const Qp={kernelName:Xo,backendName:"wasm",kernelFunc:Yp};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jp(s){const{inputs:e,backend:a,attrs:o}=s,{axis:i}=o;if(e.length===1)return Cr({inputs:{input:e[0]},backend:a,attrs:{dim:i}});const t=e[0].shape,u=e[0].dtype;e.forEach(r=>{Qo(t,r.shape,"All tensors passed to stack must have matching shapes"),Be(u===r.dtype,()=>"All tensors passed to stack must have matching dtypes")});const c=[],p=e.map(r=>{const f=Cr({inputs:{input:r},backend:a,attrs:{dim:i}});return c.push(f),f}),h=$s({inputs:p,backend:a,attrs:{axis:i}});return c.forEach(r=>a.disposeData(r.dataId)),h}const Zp={kernelName:Yo,backendName:"wasm",kernelFunc:Jp};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let gi;function ed(s){gi=s.wasm.cwrap(ys,null,["number","array","number","number","array","array","number","number"])}function td(s){const{inputs:{x:e},backend:a,attrs:{paddings:o,constantValue:i}}=s,t=o.map((v,x)=>v[0]+e.shape[x]+v[1]);if(X(e.shape)===0)return Zs({backend:a,attrs:{shape:t,value:i,dtype:e.dtype}});const u=a.dataIdMap.get(e.dataId).id,c=a.makeOutput(t,e.dtype),h=a.dataIdMap.get(c.dataId).id,r=new Uint8Array(new Int32Array(e.shape).buffer),f=o.map(v=>v[0]),b=o.map(v=>v[1]),g=new Uint8Array(new Int32Array(f).buffer),_=new Uint8Array(new Int32Array(b).buffer);return gi(u,r,e.shape.length,be[e.dtype],g,_,i,h),c}const yi={kernelName:ys,backendName:"wasm",kernelFunc:td,setupFunc:ed};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nd=Ae(Jo);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let bi;function rd(s){bi=s.wasm.cwrap(bs,null,["number","number","number"])}function ad(s){const{inputs:e,backend:a}=s,{x:o,alpha:i}=e,t=a.dataIdMap.get(o.dataId).id,u=a.dataIdMap.get(i.dataId).id;let c=t;const p=o;let h=p;p.dtype!=="float32"&&(h=yn({backend:a,inputs:{x:o},attrs:{dtype:"float32"}}),c=a.dataIdMap.get(h.dataId).id);const r=a.makeOutput(o.shape,"float32"),f=a.dataIdMap.get(r.dataId).id;return bi(c,u,f),p.dtype!=="float32"&&a.disposeData(h.dataId),r}const sd={kernelName:bs,backendName:"wasm",setupFunc:rd,kernelFunc:ad};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let _i;function id(s){_i=s.wasm.cwrap(_s,null,["number","number","number","number"])}function od(s){const{backend:e,inputs:a,attrs:o}=s,{axis:i,keepDims:t}=o,{x:u}=a,c=e.dataIdMap.get(u.dataId).id;let p=c,h=u;const{transposed:r,axes:f,originalAxes:b,inputWasTransposed:g}=Ct(u,i,e);let _=f;if(g){const I=e.dataIdMap.get(r.dataId).id;I!==c&&(h=r,p=I,_=mn(_.length,h.shape.length))}kt("prod",_,h.shape.length);const[v,x]=Qt(h.shape,_),S=X(x),k=e.makeOutput(v,h.dtype);if(X(h.shape)!==0){const I=e.dataIdMap.get(k.dataId).id;_i(p,S,be[k.dtype],I)}if(g&&e.disposeData(r.dataId),t){const I=Jt(k.shape,b);k.shape=I}return k}const ud={kernelName:_s,backendName:"wasm",setupFunc:id,kernelFunc:od};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ld=s=>{const{backend:e,attrs:a}=s,{start:o,stop:i,step:t,dtype:u}=a,c=Ll(o,i,t,u),p=e.makeOutput([c.length],u);return e.typedArrayFromHeap(p).set(c),p},cd={kernelName:Zo,backendName:"wasm",kernelFunc:ld};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pd=Ae(eu);/**
 * @license
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const dd=we(tu);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hd=we(nu);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fd=we(ru);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let vi;function md(s){vi=s.wasm.cwrap(vs,null,["number","number","number","number","number","number","number","number","number","number"])}function gd(s){const{backend:e,inputs:a,attrs:o}=s,{images:i}=a,{alignCorners:t,halfPixelCenters:u,size:c}=o,[p,h]=c,[r,f,b,g]=i.shape,_=[r,p,h,g];let v=e.dataIdMap.get(i.dataId),x;v.dtype!=="float32"&&(x=yn({backend:e,inputs:{x:i},attrs:{dtype:"float32"}}),v=e.dataIdMap.get(x.dataId));const S=v.id,k=e.makeOutput(_,"float32");if(X(i.shape)===0)return k;const I=e.dataIdMap.get(k.dataId).id;return vi(S,r,f,b,g,p,h,t?1:0,u?1:0,I),x!=null&&e.disposeData(x.dataId),k}const yd={kernelName:vs,backendName:"wasm",setupFunc:md,kernelFunc:gd};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let wi;function bd(s){wi=s.wasm.cwrap(ws,null,["number","number","number","number","number","number","number","number","number","number"])}function _d(s){const{backend:e,inputs:a,attrs:o}=s,{images:i}=a,{alignCorners:t,halfPixelCenters:u,size:c}=o,[p,h]=c,[r,f,b,g]=i.shape,_=[r,p,h,g],v=e.makeOutput(_,"float32");if(X(i.shape)===0)return v;let x=e.dataIdMap.get(i.dataId),S;x.dtype!=="float32"&&(S=yn({backend:e,inputs:{x:i},attrs:{dtype:"float32"}}),x=e.dataIdMap.get(S.dataId));const k=x.id,I=e.dataIdMap.get(v.dataId).id;return wi(k,r,f,b,g,p,h,t?1:0,u?1:0,I),S!=null&&e.disposeData(S.dataId),v}const vd={kernelName:ws,backendName:"wasm",setupFunc:bd,kernelFunc:_d};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let ki;function wd(s){ki=s.wasm.cwrap(ks,null,["number","array","number","array","number","number"])}function kd(s){const{inputs:e,backend:a,attrs:o}=s,{x:i}=e,{dims:t}=o,u=On(t,i.shape);if(i.shape.length===0)return ar({inputs:{x:i},backend:a});const c=a.makeOutput(i.shape,i.dtype),p=a.dataIdMap.get(i.dataId).id,h=a.dataIdMap.get(c.dataId).id,r=new Uint8Array(new Int32Array(u).buffer),f=new Uint8Array(new Int32Array(i.shape).buffer);ki(p,r,u.length,f,i.shape.length,h);const b=Le({inputs:{x:c},attrs:{shape:i.shape},backend:a});return a.disposeData(c.dataId),b}const Id={kernelName:ks,backendName:"wasm",kernelFunc:kd,setupFunc:wd};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Ii;function xd(s){Ii=s.wasm.cwrap(Is,null,["number","number","number","number","number","number","number","number","array","number","number"])}function Sd(s){const{inputs:e,backend:a,attrs:o}=s,{image:i}=e,{radians:t,fillValue:u,center:c}=o,p=a.makeOutput(i.shape,i.dtype),h=a.dataIdMap.get(i.dataId).id,r=a.dataIdMap.get(p.dataId).id,[f,b,g,_]=i.shape,[v,x]=au(c,b,g),S=u===0,k=255,I=typeof u=="number"?[u,u,u,S?0:k]:[...u,k],R=new Uint8Array(new Int32Array(I).buffer);return Ii(h,f,b,g,_,t,v,x,R,I.length,r),p}const Md={kernelName:Is,backendName:"wasm",kernelFunc:Sd,setupFunc:xd};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ad=we(su);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Td=we(iu);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let xi;function Rd(s){xi=s.wasm.cwrap(xs,null,["number","number","number","number","number","number","array","number","number"])}function Fd(s){const{backend:e,inputs:a,attrs:o}=s,{indices:i,updates:t}=a,{shape:u}=o,c=e.makeOutput(u,t.dtype);if(X(u)===0)return c;const{sliceRank:p,numUpdates:h,sliceSize:r,strides:f,outputSize:b}=ou(t,i,u),_=e.dataIdMap.get(i.dataId).id,x=e.dataIdMap.get(t.dataId).id,S=new Uint8Array(new Int32Array(f).buffer),k=e.dataIdMap.get(c.dataId).id;return xi(_,x,be[t.dtype],p,h,r,S,b,k),c}const Ed={kernelName:xs,backendName:"wasm",setupFunc:Rd,kernelFunc:Fd};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Si;function Cd(s){Si=s.wasm.cwrap("SelectV2",null,["number","number","number","number","number"])}function Nd(s){const{inputs:e,backend:a}=s,{condition:o,t:i,e:t}=e,u=a.dataIdMap.get(o.dataId).id,c=a.dataIdMap.get(i.dataId).id,p=a.dataIdMap.get(t.dataId).id,h=a.makeOutput(i.shape,i.dtype),r=a.dataIdMap.get(h.dataId).id,f=o.shape.length,b=i.shape.length,g=f===0||f>1||b===1?1:X(i.shape.slice(1));return Si(u,c,p,g,r),h}const Od={kernelName:uu,backendName:"wasm",kernelFunc:Nd,setupFunc:Cd};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Mi;function Pd(s){Mi=s.wasm.cwrap(lu,null,["number","number"])}function Dd(s){const{backend:e,inputs:{x:a}}=s,o=e.dataIdMap.get(a.dataId).id,i=e.makeOutput(a.shape,a.dtype),t=e.dataIdMap.get(i.dataId).id;return X(i.shape)===0||Mi(o,t),i}const Bd={kernelName:"Sigmoid",backendName:"wasm",setupFunc:Pd,kernelFunc:Dd};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ld=we(cu);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Ai;function jd(s){Ai=s.wasm.cwrap(Ss,null,["number","number","number","number"])}function Wd(s){const{backend:e,inputs:{logits:a},attrs:{dim:o}}=s,i=e.dataIdMap.get(a.dataId).id,t=e.makeOutput(a.shape,a.dtype),u=e.dataIdMap.get(t.dataId).id,c=a.shape[o],p=X(a.shape)/c;return X(t.shape)===0||Ai(i,u,c,p),t}const Hd={kernelName:Ss,backendName:"wasm",setupFunc:jd,kernelFunc:Wd};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vd(s){const{inputs:e,backend:a,attrs:o}=s,{x:i}=e,{blockShape:t,paddings:u}=o,c=X(t),p=[[0,0]];p.push(...u);for(let E=1+t.length;E<i.shape.length;++E)p.push([0,0]);const h=yi.kernelFunc({inputs:{x:i},backend:a,attrs:{paddings:p,constantValue:0}}),r=Ua(h.shape,t,c,!1),f=za(r.length,t.length,!1),b=$a(h.shape,t,c,!1),v=Le({inputs:{x:h},backend:a,attrs:{shape:r}}),k=Et({inputs:{x:v},backend:a,attrs:{perm:f}}),N=Le({inputs:{x:k},backend:a,attrs:{shape:b}});return a.disposeData(h.dataId),a.disposeData(v.dataId),a.disposeData(k.dataId),N}const Ud={kernelName:pu,backendName:"wasm",kernelFunc:Vd};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Ti;function zd(s){Ti=s.wasm.cwrap("SparseFillEmptyRows","number",["number","number","number","number","number","number","number","number","number","number","number","number"])}function $d(s){const{backend:e,inputs:a}=s,{indices:o,values:i,denseShape:t,defaultValue:u}=a,c=o.shape[0],p=o.shape[1],h=e.readSync(t.dataId)[0],r=[c+h,p],f=e.dataIdMap.get(o.dataId).id,b=e.dataIdMap.get(i.dataId).id,g=e.dataIdMap.get(u.dataId).id,_=e.makeOutput(r,o.dtype),v=e.dataIdMap.get(_.dataId).id,x=e.makeOutput(r.slice(0,1),i.dtype),S=e.dataIdMap.get(x.dataId).id,k=e.makeOutput([h],"bool"),I=e.dataIdMap.get(k.dataId).id,R=e.makeOutput([c],o.dtype),N=e.dataIdMap.get(R.dataId).id,E=e.makeOutput([4],"int32"),O=e.dataIdMap.get(E.dataId).id,j=Ti(f,b,be[i.dtype],c,h,p,g,v,S,I,N,O),B=e.readSync(E.dataId);let D;switch(B[0]){case 1:{D=mu(B[1]);break}case 2:{D=fu(B[1],B[2]);break}case 3:D=hu(B[1],B[2],B[3]);break;default:D=""}if(e.disposeData(E.dataId),D)throw e.disposeData(_.dataId),e.disposeData(x.dataId),e.disposeData(k.dataId),e.disposeData(R.dataId),new Error(D);let V=_,q=x;return j!==r[0]&&(V=Yt({inputs:{x:_},attrs:{begin:0,size:[j,p]},backend:e}),q=Yt({inputs:{x},attrs:{begin:0,size:j},backend:e}),e.disposeData(_.dataId),e.disposeData(x.dataId)),[V,q,k,R]}const Gd={kernelName:du,backendName:"wasm",setupFunc:zd,kernelFunc:$d};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Ri;function qd(s){Ri=s.wasm.cwrap(Ms,null,["number","number","number","number","number","number","number"])}function Kd(s){const{backend:e,inputs:a}=s,{inputIndices:o,inputShape:i,newShape:t}=a;if(o.shape.length!==2)throw new Error(`Input indices should be a matrix but received shape
        ${o.shape}`);if(i.shape.length!==1)throw new Error(`Input shape should be a vector but received shape
        ${i.shape}`);if(t.shape.length!==1)throw new Error(`Target shape should be a vector but received shape ${t.shape}`);const u=e.dataIdMap.get(o.dataId).id,c=e.dataIdMap.get(i.dataId).id,p=e.dataIdMap.get(t.dataId).id,h=o.shape[0],r=X(t.shape),f=e.makeOutput([h,r],o.dtype),b=e.dataIdMap.get(f.dataId).id,g=e.makeOutput([r],t.dtype),_=e.dataIdMap.get(g.dataId).id,v=e.makeOutput([3],"int32"),x=e.dataIdMap.get(v.dataId).id;Ri(u,c,p,h,b,_,x);const S=e.readSync(v.dataId);let k;switch(S[0]){case 0:{k=_u(S[1],S[2]);break}case 1:{k=bu(S[1],S[2]);break}case 2:k=vu();break;case 3:{const I=Array.from(e.readSync(i.dataId)),R=Array.from(e.readSync(g.dataId));k=yu(I,R);break}case 4:{const I=Array.from(e.readSync(i.dataId)),R=Array.from(e.readSync(g.dataId));k=gu(I,R);break}default:k=""}if(e.disposeData(v.dataId),k)throw e.disposeData(f.dataId),e.disposeData(g.dataId),new Error(k);return[f,g]}const Xd={kernelName:Ms,backendName:"wasm",setupFunc:qd,kernelFunc:Kd};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Fi;function Ei(s){Fi=s.wasm.cwrap("SparseSegmentReduction",null,["number","number","number","number","number","number","number","number","number"])}function Ci(s,e){const{backend:a,inputs:o}=s,{data:i,indices:t,segmentIds:u}=o,c=t.shape[0],p=a.readSync(u.dataId,c-1,c)[0],r=c>0?p+1:0;if(r<0)throw new Error(qr());const f=i.shape.slice();f[0]=r;const b=a.dataIdMap.get(i.dataId).id,g=a.dataIdMap.get(t.dataId).id,_=a.dataIdMap.get(u.dataId).id,v=a.makeOutput(f,i.dtype),x=a.dataIdMap.get(v.dataId).id,S=a.makeOutput([4],"int32"),k=a.dataIdMap.get(S.dataId).id;Fi(b,be[i.dtype],i.shape[0],g,_,x,k,e,0);const I=a.readSync(S.dataId);let R;switch(I[0]){case 0:{R=qr();break}case 1:{R=Iu();break}case 2:R=ku(I[1],I[2]);break;case 3:R=wu(I[1],I[2],I[3]);break;default:R=""}if(a.disposeData(S.dataId),R)throw a.disposeData(v.dataId),new Error(R);return v}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yd(s){return Ci(s,!0)}const Qd={kernelName:xu,backendName:"wasm",setupFunc:Ei,kernelFunc:Yd};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jd(s){return Ci(s,!1)}const Zd={kernelName:Su,backendName:"wasm",setupFunc:Ei,kernelFunc:Jd};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eh(s){const{inputs:e,attrs:a,backend:o}=s,{x:i}=e,{numOrSizeSplits:t,axis:u}=a,c=On(u,i.shape)[0],p=Au(i,t,c),h=new Array(i.shape.length).fill(0),r=i.shape.slice();return p.map(f=>{const b=[...r];b[c]=f;const g=Yt({inputs:{x:i},attrs:{begin:h,size:b},backend:o});return h[c]+=f,g})}const th={kernelName:Mu,backendName:"wasm",kernelFunc:eh};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nh=we(Tu);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rh=we(Ru);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ah=Ae(Fu);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Ni;function sh(s){Ni=s.wasm.cwrap(As,null,["number","number","number","number"])}function ih(s){const{backend:e,inputs:a,attrs:o}=s,{alpha:i}=o,{x:t}=a,u=e.dataIdMap.get(t.dataId).id,c=e.makeOutput(t.shape,t.dtype),p=e.dataIdMap.get(c.dataId).id;return Ni(u,i,be[t.dtype],p),c}const oh={kernelName:As,backendName:"wasm",setupFunc:sh,kernelFunc:ih};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Oi;function uh(s){Oi=s.wasm.cwrap(Ts,null,["number","array","number","array","array","array","array","array","number","number"])}function lh(s){const{backend:e,inputs:a,attrs:o}=s,{x:i}=a,{begin:t,end:u,strides:c,beginMask:p,endMask:h,ellipsisMask:r,newAxisMask:f,shrinkAxisMask:b}=o,{finalShapeSparse:g,finalShape:_,isIdentity:v,sliceDim0:x,isSimpleSlice:S,begin:k,end:I,strides:R}=Eu(i.shape,t,u,c,p,h,r,f,b);let N;if(v)N=Le({inputs:{x:i},backend:e,attrs:{shape:_}});else if(x||S){Be(i.shape.length>=1,()=>`Input must have rank at least 1, got: ${i.shape.length}`);const E=Cu(k,I,R),O=Yt({inputs:{x:i},backend:e,attrs:{begin:k,size:E}});N=Le({inputs:{x:O},backend:e,attrs:{shape:_}}),e.disposeData(O.dataId)}else{const E=e.makeOutput(g,"float32"),O=e.dataIdMap.get(i.dataId).id,j=new Uint8Array(new Int32Array(Ge(i.shape)).buffer),B=new Uint8Array(new Int32Array(k).buffer),D=new Uint8Array(new Int32Array(I).buffer),V=new Uint8Array(new Int32Array(R).buffer),q=new Uint8Array(new Int32Array(g).buffer),$=new Uint8Array(new Int32Array(Ge(g)).buffer),Y=e.dataIdMap.get(E.dataId).id;Oi(O,j,i.shape.length,B,D,V,q,$,g.length,Y),N=Le({inputs:{x:E},backend:e,attrs:{shape:_}}),e.disposeData(E.dataId)}return N}const ch={kernelName:Ts,backendName:"wasm",setupFunc:uh,kernelFunc:lh};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ph(s){const{backend:e,inputs:a,attrs:o}=s,{data:i,dataSplits:t}=a,{separator:u,nGramWidths:c,leftPad:p,rightPad:h,padWidth:r,preserveShortSequences:f}=o,b=e.readSync(i.dataId),g=e.readSync(t.dataId),[_,v]=Wl(b,g,u,c,p,h,r,f),x=e.makeOutput([_.length],"string"),S=e.dataIdMap.get(x.dataId);S.stringBytes=_;const k=e.makeOutput(t.shape,"int32");return e.typedArrayFromHeap(k).set(v),[x,k]}const dh={kernelName:Nu,backendName:"wasm",kernelFunc:ph};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hh(s){const{backend:e,inputs:a,attrs:o}=s,{input:i,delimiter:t}=a,{skipEmpty:u}=o,c=e.readSync(i.dataId),p=e.readSync(t.dataId),[h,r,f]=Vl(c,p[0],u),b=r.length,g=e.makeOutput([b,2],"int32");e.typedArrayFromHeap(g).set(h);const v=e.makeOutput([b],"string"),x=e.dataIdMap.get(v.dataId);x.stringBytes=r;const S=e.makeOutput([2],"int32");return e.typedArrayFromHeap(S).set(f),[g,v,S]}const fh={kernelName:Ou,backendName:"wasm",kernelFunc:hh};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mh(s){const{backend:e,inputs:a,attrs:o}=s,{input:i}=a,{numBuckets:t}=o,u=e.readSync(i.dataId),c=Ul(u,t),p=e.makeOutput(i.shape,"int32");return e.typedArrayFromHeap(p).set(c),p}const gh={kernelName:Pu,backendName:"wasm",kernelFunc:mh};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yh=Ae(Du);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Pi;function bh(s){Pi=s.wasm.cwrap(Rs,null,["number","number","number","number"])}function _h(s){const{backend:e,inputs:a,attrs:o}=s,{axis:i,keepDims:t}=o,{x:u}=a,c=e.dataIdMap.get(u.dataId).id;let p=c,h=u;const{transposed:r,axes:f,originalAxes:b,inputWasTransposed:g}=Ct(u,i,e);let _=f;if(g){const I=e.dataIdMap.get(r.dataId).id;I!==c&&(h=r,p=I,_=mn(_.length,h.shape.length))}kt("sum",_,h.shape.length);const[v,x]=Qt(h.shape,_),S=X(x),k=e.makeOutput(v,h.dtype);if(X(h.shape)!==0){const I=e.dataIdMap.get(k.dataId).id;Pi(p,S,be[k.dtype],I)}if(g&&e.disposeData(r.dataId),t){const I=Jt(k.shape,b);k.shape=I}return k}const vh={kernelName:Rs,backendName:"wasm",setupFunc:bh,kernelFunc:_h};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wh=we(Bu);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kh=we(Lu);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Di;function Ih(s){Di=s.wasm.cwrap(Fs,null,["number","array","number","array","number","number"])}function xh(s){const{inputs:e,backend:a,attrs:o}=s,{x:i}=e,t=a.dataIdMap.get(i.dataId).id,{reps:u}=o,c=new Array(i.shape.length);for(let b=0;b<c.length;b++)c[b]=i.shape[b]*u[b];const p=new Uint8Array(new Int32Array(i.shape).buffer),h=new Uint8Array(new Int32Array(c).buffer),r=a.makeOutput(c,i.dtype),f=a.dataIdMap.get(r.dataId).id;return Di(t,p,i.shape.length,h,c.length,be[r.dtype],f),r}const Sh={kernelName:Fs,backendName:"wasm",setupFunc:Ih,kernelFunc:xh};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Bi;function Mh(s){Bi=s.wasm.cwrap(Es,null,["number","array","number","number","number","bool","number","number"])}const Ah=({inputs:s,backend:e,attrs:a})=>{const{x:o}=s,{k:i,sorted:t}=a,u=e.dataIdMap.get(o.dataId).id,c=new Uint8Array(new Int32Array(o.shape).buffer),p=o.shape.slice();p[p.length-1]=i;const h=e.makeOutput(p,o.dtype),r=e.dataIdMap.get(h.dataId).id,f=e.makeOutput(p,"int32"),b=e.dataIdMap.get(f.dataId).id;return Bi(u,c,o.shape.length,be[o.dtype],i,t,r,b),[h,f]},Th={kernelName:Es,backendName:"wasm",setupFunc:Mh,kernelFunc:Ah};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Li;function Rh(s){Li=s.wasm.cwrap(Cs,null,["number","number","bool","number","number","number","number","number","number","array","number","array","number","number","number","number","number"])}function Fh(s){const{backend:e,inputs:a,attrs:o}=s,{image:i,transforms:t}=a,{interpolation:u,fillMode:c,fillValue:p,outputShape:h}=o,[r,f,b,g]=i.shape,[_,v]=h??[f,b],x=[r,_,v,g],S=new Uint8Array(new Int32Array(Ge(i.shape)).buffer),k=new Uint8Array(new Int32Array(Ge(x)).buffer),I=e.makeOutput(x,i.dtype),R=e.dataIdMap.get(I.dataId).id,E=e.dataIdMap.get(i.dataId).id,j=e.dataIdMap.get(t.dataId).id,B=u==="nearest"?1:2;let D;switch(c){case"constant":D=1;break;case"reflect":D=2;break;case"wrap":D=3;break;case"nearest":D=4;break;default:D=1;break}return Li(E,j,t.shape[0]>1,r,_,v,g,b,f,S,i.shape.length-1,k,x.length-1,B,D,p,R),I}const Eh={kernelName:Cs,backendName:"wasm",setupFunc:Rh,kernelFunc:Fh};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ch(s){const{inputs:e,backend:a,attrs:o}=s,{value:i}=e;let{axis:t}=o;t<0&&(t+=i.shape.length);const u=i.shape[t],c=i.shape.length,p=new Array(c-1);let h=0;for(let g=0;g<c;g++)g!==t&&(p[h++]=i.shape[g]);const r=new Array(u),f=new Array(c).fill(0),b=i.shape.slice();b[t]=1;for(let g=0;g<r.length;g++)f[t]=g,r[g]=Yt({inputs:{x:i},attrs:{begin:f,size:b},backend:a});return r.map(({dataId:g,dtype:_})=>({dataId:g,dtype:_,shape:p}))}const Nh={kernelName:ju,backendName:"wasm",kernelFunc:Ch};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oh(s){const{inputs:{x:e},backend:a}=s,o=a.makeOutput(e.shape,e.dtype);return a.typedArrayFromHeap(o).fill(0),o}const Ph={kernelName:Wu,backendName:"wasm",kernelFunc:Oh};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Dh=[cl,pl,dl,ml,Il,Ml,Rl,Cl,Dl,Xl,Yl,Ql,ec,tc,ac,oc,uc,lc,dc,mc,bc,wc,xc,Sc,Ac,Tc,Rc,Fc,Nc,Oc,Pc,Lc,Hc,zc,qc,Yc,Jc,ep,gl,tp,ap,ip,up,lp,pp,dp,fp,gp,_p,vp,Ip,Mp,Rp,Fp,Np,Op,Pp,Lp,Hp,zp,Gp,Xp,Qp,Zp,yi,nd,sd,ud,cd,pd,dd,hd,fd,Nl,yd,vd,Id,Md,Ad,Td,Ed,Od,Bd,Ld,ql,Hd,Ud,Gd,Xd,Qd,Zd,th,nh,rh,ah,oh,ch,dh,fh,gh,yh,vh,wh,kh,Sh,Th,Eh,vl,Nh,Ph];for(const s of Dh)Hu(s);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Or=Jn();Or.registerFlag("WASM_HAS_SIMD_SUPPORT",async()=>{try{return WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,10,9,1,7,0,65,0,253,15,26,11]))}catch{return!1}});Or.registerFlag("WASM_HAS_MULTITHREAD_SUPPORT",async()=>{if(Or.get("IS_NODE"))return!1;try{return new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)),WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,5,4,1,3,1,1,10,11,1,9,0,65,0,254,16,2,0,26,11]))}catch{return!1}});var er={},Bh={get exports(){return er},set exports(s){er=s}};(function(s,e){var a=(()=>{var o=typeof document<"u"&&document.currentScript?document.currentScript.src:void 0;return typeof __filename<"u"&&(o=o||__filename),function(i){i=i||{};function t(){return ae.buffer!=ge&&Xe(ae.buffer),at}function u(){return ae.buffer!=ge&&Xe(ae.buffer),Nt}function c(){return ae.buffer!=ge&&Xe(ae.buffer),xt}function p(){return ae.buffer!=ge&&Xe(ae.buffer),Ke}function h(){return ae.buffer!=ge&&Xe(ae.buffer),Ot}var r=typeof i<"u"?i:{},f,b;r.ready=new Promise(function(w,T){f=w,b=T});var g;typeof process<"u"&&process.listeners&&(g={uncaughtException:process.listeners("uncaughtException"),unhandledRejection:process.listeners("unhandledRejection")});var _=Object.assign({},r),v=(w,T)=>{throw T},x=typeof window=="object",S=typeof importScripts=="function",k=typeof process=="object"&&typeof process.versions=="object"&&typeof process.versions.node=="string",I=r.ENVIRONMENT_IS_PTHREAD||!1,R="";function N(w){return r.locateFile?r.locateFile(w,R):R+w}var E,O,j;function B(w){if(w instanceof Je)return;Q("exiting due to exception: "+w)}if(k){var D=zt,V=zt;S?R=V.dirname(R)+"/":R=__dirname+"/",E=(T,n)=>(T=vt(T)?new URL(T):V.normalize(T),D.readFileSync(T,n?void 0:"utf8")),j=T=>{var n=E(T,!0);return n.buffer||(n=new Uint8Array(n)),n},O=(T,n,l)=>{T=vt(T)?new URL(T):V.normalize(T),D.readFile(T,function(d,m){d?l(d):n(m.buffer)})},process.argv.length>1&&process.argv[1].replace(/\\/g,"/"),process.argv.slice(2),process.on("uncaughtException",function(T){if(!(T instanceof Je))throw T}),process.on("unhandledRejection",function(T){throw T}),v=(T,n)=>{if(it())throw process.exitCode=T,n;B(n),process.exit(T)},r.inspect=function(){return"[Emscripten Module object]"};let w;try{w=zt}catch(T){throw console.error('The "worker_threads" module is not supported in this node.js build - perhaps a newer version is needed?'),T}Tn.Worker=w.Worker}else(x||S)&&(S?R=self.location.href:typeof document<"u"&&document.currentScript&&(R=document.currentScript.src),typeof o<"u"&&o&&(R=o),R.indexOf("blob:")!==0?R=R.substr(0,R.replace(/[?#].*/,"").lastIndexOf("/")+1):R="",k||(E=w=>{var T=new XMLHttpRequest;return T.open("GET",w,!1),T.send(null),T.responseText},S&&(j=w=>{var T=new XMLHttpRequest;return T.open("GET",w,!1),T.responseType="arraybuffer",T.send(null),new Uint8Array(T.response)}),O=(w,T,n)=>{var l=new XMLHttpRequest;l.open("GET",w,!0),l.responseType="arraybuffer",l.onload=()=>{if(l.status==200||l.status==0&&l.response){T(l.response);return}n()},l.onerror=n,l.send(null)}));k&&typeof performance>"u"&&(Tn.performance=zt.performance);var q=console.log.bind(console),$=console.warn.bind(console);k&&(q=w=>D.writeSync(1,w+`
`),$=w=>D.writeSync(2,w+`
`));var Y=r.print||q,Q=r.printErr||$;Object.assign(r,_),_=null,r.arguments&&r.arguments,r.thisProgram&&r.thisProgram,r.quit&&(v=r.quit);var ee;r.wasmBinary&&(ee=r.wasmBinary);var ce=r.noExitRuntime||!0;typeof WebAssembly!="object"&&Qe("no native wasm support detected");var ae,xe,Te=!1,Ee;function qe(w,T){w||Qe(T)}var ke=typeof TextDecoder<"u"?new TextDecoder("utf8"):void 0;function yt(w,T,n){for(var l=T+n,d=T;w[d]&&!(d>=l);)++d;if(d-T>16&&w.buffer&&ke)return ke.decode(w.buffer instanceof SharedArrayBuffer?w.slice(T,d):w.subarray(T,d));for(var m="";T<d;){var y=w[T++];if(!(y&128)){m+=String.fromCharCode(y);continue}var M=w[T++]&63;if((y&224)==192){m+=String.fromCharCode((y&31)<<6|M);continue}var A=w[T++]&63;if((y&240)==224?y=(y&15)<<12|M<<6|A:y=(y&7)<<18|M<<12|A<<6|w[T++]&63,y<65536)m+=String.fromCharCode(y);else{var P=y-65536;m+=String.fromCharCode(55296|P>>10,56320|P&1023)}}return m}function bt(w,T){return w?yt(u(),w,T):""}function nt(w,T,n,l){if(!(l>0))return 0;for(var d=n,m=n+l-1,y=0;y<w.length;++y){var M=w.charCodeAt(y);if(M>=55296&&M<=57343){var A=w.charCodeAt(++y);M=65536+((M&1023)<<10)|A&1023}if(M<=127){if(n>=m)break;T[n++]=M}else if(M<=2047){if(n+1>=m)break;T[n++]=192|M>>6,T[n++]=128|M&63}else if(M<=65535){if(n+2>=m)break;T[n++]=224|M>>12,T[n++]=128|M>>6&63,T[n++]=128|M&63}else{if(n+3>=m)break;T[n++]=240|M>>18,T[n++]=128|M>>12&63,T[n++]=128|M>>6&63,T[n++]=128|M&63}}return T[n]=0,n-d}function It(w,T,n){return nt(w,u(),T,n)}var ge,at,Nt,xt,Ke,Ot;I&&(ge=r.buffer);function Xe(w){ge=w,r.HEAP8=at=new Int8Array(w),r.HEAP16=new Int16Array(w),r.HEAP32=xt=new Int32Array(w),r.HEAPU8=Nt=new Uint8Array(w),r.HEAPU16=new Uint16Array(w),r.HEAPU32=Ke=new Uint32Array(w),r.HEAPF32=new Float32Array(w),r.HEAPF64=Ot=new Float64Array(w)}var st=r.INITIAL_MEMORY||16777216;if(I)ae=r.wasmMemory,ge=r.buffer;else if(r.wasmMemory)ae=r.wasmMemory;else if(ae=new WebAssembly.Memory({initial:st/65536,maximum:32768,shared:!0}),!(ae.buffer instanceof SharedArrayBuffer))throw Q("requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag"),k&&Q("(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and/or recent version)"),Error("bad memory");ae&&(ge=ae.buffer),st=ge.byteLength,Xe(ge);var he,je=[],Zt=[],Pt=[];function it(){return ce}function Se(){if(r.preRun)for(typeof r.preRun=="function"&&(r.preRun=[r.preRun]);r.preRun.length;)Bt(r.preRun.shift());se(je)}function ot(){I||se(Zt)}function Dt(){if(!I){if(r.postRun)for(typeof r.postRun=="function"&&(r.postRun=[r.postRun]);r.postRun.length;)ut(r.postRun.shift());se(Pt)}}function Bt(w){je.unshift(w)}function We(w){Zt.unshift(w)}function ut(w){Pt.unshift(w)}var Ye=0,lt=null;function _t(w){Ye++,r.monitorRunDependencies&&r.monitorRunDependencies(Ye)}function en(w){if(Ye--,r.monitorRunDependencies&&r.monitorRunDependencies(Ye),Ye==0&&lt){var T=lt;lt=null,T()}}function Qe(w){r.onAbort&&r.onAbort(w),w="Aborted("+w+")",Q(w),Te=!0,Ee=1,w+=". Build with -sASSERTIONS for more info.";var T=new WebAssembly.RuntimeError(w);throw b(T),T}var bn="data:application/octet-stream;base64,";function ct(w){return w.startsWith(bn)}function vt(w){return w.startsWith("file://")}var _e;_e="tfjs-backend-wasm-threaded-simd.wasm",ct(_e)||(_e=N(_e));function Lt(w){try{if(w==_e&&ee)return new Uint8Array(ee);if(j)return j(w);throw"both async and sync fetching of the wasm failed"}catch(T){Qe(T)}}function St(){if(!ee&&(x||S)){if(typeof fetch=="function"&&!vt(_e))return fetch(_e,{credentials:"same-origin"}).then(function(w){if(!w.ok)throw"failed to load wasm binary file at '"+_e+"'";return w.arrayBuffer()}).catch(function(){return Lt(_e)});if(O)return new Promise(function(w,T){O(_e,function(n){w(new Uint8Array(n))},T)})}return Promise.resolve().then(function(){return Lt(_e)})}function pt(){var w={env:Sn,wasi_snapshot_preview1:Sn};function T(y,M){var A=y.exports;if(r.asm=A,Ze(r.asm._emscripten_tls_init),he=r.asm.__indirect_function_table,We(r.asm.__wasm_call_ctors),xe=M,!I){var P=W.unusedWorkers.length;W.unusedWorkers.forEach(function(C){W.loadWasmModuleToWorker(C,function(){--P||en()})})}}I||_t();function n(y){T(y.instance,y.module)}function l(y){return St().then(function(M){return WebAssembly.instantiate(M,w)}).then(function(M){return M}).then(y,function(M){Q("failed to asynchronously prepare wasm: "+M),Qe(M)})}function d(){return!ee&&typeof WebAssembly.instantiateStreaming=="function"&&!ct(_e)&&!vt(_e)&&!k&&typeof fetch=="function"?fetch(_e,{credentials:"same-origin"}).then(function(y){var M=WebAssembly.instantiateStreaming(y,w);return M.then(n,function(A){return Q("wasm streaming compile failed: "+A),Q("falling back to ArrayBuffer instantiation"),l(n)})}):l(n)}if(r.instantiateWasm)try{var m=r.instantiateWasm(w,T);return m}catch(y){Q("Module.instantiateWasm callback failed with error: "+y),b(y)}return d().catch(b),{}}var jt={};function Je(w){this.name="ExitStatus",this.message="Program terminated with exit("+w+")",this.status=w}function Mt(w){var T=W.pthreads[w];delete W.pthreads[w],T.terminate(),sn(w),W.runningWorkers.splice(W.runningWorkers.indexOf(T),1),T.pthread_ptr=0}function Wt(w){var T=W.pthreads[w];T.postMessage({cmd:"cancel"})}function dt(w){var T=W.pthreads[w];qe(T),W.returnWorkerToPool(T)}function ht(w){var T=W.getNewWorker();if(!T)return 6;W.runningWorkers.push(T),W.pthreads[w.pthread_ptr]=T,T.pthread_ptr=w.pthread_ptr;var n={cmd:"run",start_routine:w.startRoutine,arg:w.arg,pthread_ptr:w.pthread_ptr};return T.runPthread=()=>{k&&T.ref(),T.postMessage(n,w.transferList),delete T.runPthread},T.loaded&&T.runPthread(),0}function Re(w){if(I)return wt(1,1,w);Ee=w,it()||(W.terminateAllThreads(),r.onExit&&r.onExit(w),Te=!0),v(w,new Je(w))}function F(w,T){if(Ee=w,!T&&I)throw te(w),"unwind";Re(w)}var L=F;function U(w){if(w instanceof Je||w=="unwind")return Ee;v(1,w)}var W={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init:function(){I?W.initWorker():W.initMainThread()},initMainThread:function(){for(var w=8;w--;)W.allocateUnusedWorker()},initWorker:function(){ce=!1},setExitStatus:function(w){Ee=w},terminateAllThreads:function(){for(var w of Object.values(W.pthreads))W.returnWorkerToPool(w);for(var w of W.unusedWorkers)w.terminate();W.unusedWorkers=[]},returnWorkerToPool:function(w){var T=w.pthread_ptr;delete W.pthreads[T],W.unusedWorkers.push(w),W.runningWorkers.splice(W.runningWorkers.indexOf(w),1),w.pthread_ptr=0,k&&w.unref(),sn(T)},receiveObjectTransfer:function(w){},threadInitTLS:function(){W.tlsInitFunctions.forEach(w=>w())},loadWasmModuleToWorker:function(w,T){w.onmessage=m=>{var y=m.data,M=y.cmd;if(w.pthread_ptr&&(W.currentProxiedOperationCallerThread=w.pthread_ptr),y.targetThread&&y.targetThread!=an()){var A=W.pthreads[y.targetThread];A?A.postMessage(y,y.transferList):Q('Internal error! Worker sent a message "'+M+'" to target pthread '+y.targetThread+", but that thread no longer exists!"),W.currentProxiedOperationCallerThread=void 0;return}M==="processProxyingQueue"?_n(y.queue):M==="spawnThread"?ht(y):M==="cleanupThread"?dt(y.thread):M==="killThread"?Mt(y.thread):M==="cancelThread"?Wt(y.thread):M==="loaded"?(w.loaded=!0,k&&w.unref(),T&&T(w),w.runPthread&&w.runPthread()):M==="print"?Y("Thread "+y.threadId+": "+y.text):M==="printErr"?Q("Thread "+y.threadId+": "+y.text):M==="alert"?alert("Thread "+y.threadId+": "+y.text):y.target==="setimmediate"?w.postMessage(y):M==="callHandler"?r[y.handler](...y.args):M&&Q("worker sent an unknown command "+M),W.currentProxiedOperationCallerThread=void 0},w.onerror=m=>{var y="worker sent an error!";throw Q(y+" "+m.filename+":"+m.lineno+": "+m.message),m},k&&(w.on("message",function(m){w.onmessage({data:m})}),w.on("error",function(m){w.onerror(m)}),w.on("detachedExit",function(){}));var n=[],l=["onExit","onAbort","print","printErr"];for(var d of l)r.hasOwnProperty(d)&&n.push(d);w.postMessage({cmd:"load",handlers:n,urlOrBlob:r.mainScriptUrlOrBlob||o,wasmMemory:ae,wasmModule:xe})},allocateUnusedWorker:function(){var w,T=N("tfjs-backend-wasm-threaded-simd.worker.js");w=new Worker(T),W.unusedWorkers.push(w)},getNewWorker:function(){return W.unusedWorkers.length==0&&(W.allocateUnusedWorker(),W.loadWasmModuleToWorker(W.unusedWorkers[0])),W.unusedWorkers.pop()}};r.PThread=W;function se(w){for(;w.length>0;)w.shift()(r)}function J(){var w=an(),T=c()[w+52>>2],n=c()[w+56>>2],l=T-n;Xn(T,l),on(T)}r.establishStackSpace=J;function te(w){if(I)return wt(2,0,w);try{L(w)}catch(T){U(T)}}var Z=[];function pe(w){var T=Z[w];return T||(w>=Z.length&&(Z.length=w+1),Z[w]=T=he.get(w)),T}function Ce(w,T){var n=pe(w)(T);it()?W.setExitStatus(n):Kn(n)}r.invokeEntryPoint=Ce;function Ze(w){W.tlsInitFunctions.push(w)}function At(w){$n(w,!S,1,!x),W.threadInitTLS()}function Ht(w){I?postMessage({cmd:"cleanupThread",thread:w}):dt(w)}function ft(w,T,n,l){return I?wt(3,1,w,T,n,l):ve(w,T,n,l)}function ve(w,T,n,l){if(typeof SharedArrayBuffer>"u")return Q("Current environment does not support SharedArrayBuffer, pthreads are not available!"),6;var d=[],m=0;if(I&&(d.length===0||m))return ft(w,T,n,l);var y={startRoutine:n,pthread_ptr:w,arg:l,transferList:d};return I?(y.cmd="spawnThread",postMessage(y,d),0):ht(y)}function et(){return 65536}var Ne=!0;function $e(){return Ne}function _n(w){Atomics.store(c(),w>>2,1),an()&&qn(w),Atomics.compareExchange(c(),w>>2,1,0)}r.executeNotifiedProxyingQueue=_n;function sr(w,T,n,l){if(w==T)setTimeout(()=>_n(l));else if(I)postMessage({targetThread:w,cmd:"processProxyingQueue",queue:l});else{var d=W.pthreads[w];if(!d)return;d.postMessage({cmd:"processProxyingQueue",queue:l})}return 1}function Dn(w,T,n){return-1}function Bn(){Qe("")}function rt(w){rt.shown||(rt.shown={}),rt.shown[w]||(rt.shown[w]=1,k&&(w="warning: "+w),Q(w))}function Ln(){k||S||rt("Blocking on the main thread is very dangerous, see https://emscripten.org/docs/porting/pthreads.html#blocking-on-the-main-browser-thread")}function ir(){return Date.now()}function jn(){return 2147483648}function vn(){return jn()}var tn;k?tn=()=>{var w=process.hrtime();return w[0]*1e3+w[1]/1e6}:tn=()=>performance.timeOrigin+performance.now();function or(w,T,n){u().copyWithin(w,T,T+n)}function ur(){return k?zt.cpus().length:navigator.hardwareConcurrency}function nn(w){var T=Vt(),n=w();return on(T),n}function wt(w,T){var n=arguments.length-2,l=arguments;return nn(()=>{for(var d=n,m=un(d*8),y=m>>3,M=0;M<n;M++){var A=l[2+M];h()[y+M]=A}return Gn(w,d,m,T)})}var wn=[];function lr(w,T,n){wn.length=T;for(var l=n>>3,d=0;d<T;d++)wn[d]=h()[l+d];var m=w<0,y=m?jt[-w-1]:xn[w];return y.apply(null,wn)}function Wn(w){try{return ae.grow(w-ge.byteLength+65535>>>16),Xe(ae.buffer),1}catch{}}function Hn(w){var T=u().length;if(w=w>>>0,w<=T)return!1;var n=jn();if(w>n)return!1;let l=(A,P)=>A+(P-A%P)%P;for(var d=1;d<=4;d*=2){var m=T*(1+.2/d);m=Math.min(m,w+100663296);var y=Math.min(n,l(Math.max(w,m),65536)),M=Wn(y);if(M)return!0}return!1}function cr(){throw"unwind"}function Vn(w){return I?wt(4,1,w):52}function rn(w,T,n,l,d){return I?wt(5,1,w,T,n,l,d):70}var Un=[null,[],[]];function pr(w,T){var n=Un[w];T===0||T===10?((w===1?Y:Q)(yt(n,0)),n.length=0):n.push(T)}function zn(w,T,n,l){if(I)return wt(6,1,w,T,n,l);for(var d=0,m=0;m<n;m++){var y=p()[T>>2],M=p()[T+4>>2];T+=8;for(var A=0;A<M;A++)pr(w,u()[y+A]);d+=M}return p()[l>>2]=d,0}function kn(w){var T=r["_"+w];return T}function dr(w,T){t().set(w,T)}function hr(w,T,n,l,d){var m={string:G=>{var z=0;if(G!=null&&G!==0){var ne=(G.length<<2)+1;z=un(ne),It(G,z,ne)}return z},array:G=>{var z=un(G.length);return dr(G,z),z}};function y(G){return T==="string"?bt(G):T==="boolean"?Boolean(G):G}var M=kn(w),A=[],P=0;if(l)for(var C=0;C<l.length;C++){var H=m[n[C]];H?(P===0&&(P=Vt()),A[C]=H(l[C])):A[C]=l[C]}var K=M.apply(null,A);function re(G){return P!==0&&on(P),y(G)}return K=re(K),K}function In(w,T,n,l){n=n||[];var d=n.every(y=>y==="number"||y==="boolean"),m=T!=="string";return m&&d&&!l?kn(w):function(){return hr(w,T,n,arguments)}}W.init();var xn=[null,Re,te,ft,Vn,rn,zn],Sn={__emscripten_init_main_thread_js:At,__emscripten_thread_cleanup:Ht,__pthread_create_js:ve,_emscripten_default_pthread_stack_size:et,_emscripten_get_now_is_monotonic:$e,_emscripten_notify_task_queue:sr,_emscripten_set_offscreencanvas_size:Dn,abort:Bn,emscripten_check_blocking_allowed:Ln,emscripten_date_now:ir,emscripten_get_heap_max:vn,emscripten_get_now:tn,emscripten_memcpy_big:or,emscripten_num_logical_cores:ur,emscripten_receive_on_main_thread_js:lr,emscripten_resize_heap:Hn,emscripten_unwind_to_js_event_loop:cr,exit:L,fd_close:Vn,fd_seek:rn,fd_write:zn,memory:ae||r.wasmMemory};pt(),r.___wasm_call_ctors=function(){return(r.___wasm_call_ctors=r.asm.__wasm_call_ctors).apply(null,arguments)},r._init=function(){return(r._init=r.asm.init).apply(null,arguments)},r._init_with_threads_count=function(){return(r._init_with_threads_count=r.asm.init_with_threads_count).apply(null,arguments)},r._get_threads_count=function(){return(r._get_threads_count=r.asm.get_threads_count).apply(null,arguments)},r._register_tensor=function(){return(r._register_tensor=r.asm.register_tensor).apply(null,arguments)},r._dispose_data=function(){return(r._dispose_data=r.asm.dispose_data).apply(null,arguments)},r._dispose=function(){return(r._dispose=r.asm.dispose).apply(null,arguments)},r._Abs=function(){return(r._Abs=r.asm.Abs).apply(null,arguments)},r._Add=function(){return(r._Add=r.asm.Add).apply(null,arguments)},r._AddN=function(){return(r._AddN=r.asm.AddN).apply(null,arguments)},r._All=function(){return(r._All=r.asm.All).apply(null,arguments)},r._Any=function(){return(r._Any=r.asm.Any).apply(null,arguments)},r._ArgMax=function(){return(r._ArgMax=r.asm.ArgMax).apply(null,arguments)},r._AvgPool=function(){return(r._AvgPool=r.asm.AvgPool).apply(null,arguments)},r._BatchMatMul=function(){return(r._BatchMatMul=r.asm.BatchMatMul).apply(null,arguments)},r._Ceil=function(){return(r._Ceil=r.asm.Ceil).apply(null,arguments)},r._ClipByValue=function(){return(r._ClipByValue=r.asm.ClipByValue).apply(null,arguments)},r._Conv2D=function(){return(r._Conv2D=r.asm.Conv2D).apply(null,arguments)},r._Conv2DBackpropInput=function(){return(r._Conv2DBackpropInput=r.asm.Conv2DBackpropInput).apply(null,arguments)},r._Cos=function(){return(r._Cos=r.asm.Cos).apply(null,arguments)},r._Cosh=function(){return(r._Cosh=r.asm.Cosh).apply(null,arguments)},r._CropAndResize=function(){return(r._CropAndResize=r.asm.CropAndResize).apply(null,arguments)},r._Cumprod=function(){return(r._Cumprod=r.asm.Cumprod).apply(null,arguments)},r._Cumsum=function(){return(r._Cumsum=r.asm.Cumsum).apply(null,arguments)},r._DepthToSpace=function(){return(r._DepthToSpace=r.asm.DepthToSpace).apply(null,arguments)},r._DepthwiseConv2dNative=function(){return(r._DepthwiseConv2dNative=r.asm.DepthwiseConv2dNative).apply(null,arguments)},r._Elu=function(){return(r._Elu=r.asm.Elu).apply(null,arguments)},r._Equal=function(){return(r._Equal=r.asm.Equal).apply(null,arguments)},r._Exp=function(){return(r._Exp=r.asm.Exp).apply(null,arguments)},r._FlipLeftRight=function(){return(r._FlipLeftRight=r.asm.FlipLeftRight).apply(null,arguments)},r._Floor=function(){return(r._Floor=r.asm.Floor).apply(null,arguments)},r._FloorDiv=function(){return(r._FloorDiv=r.asm.FloorDiv).apply(null,arguments)},r._FusedBatchNorm=function(){return(r._FusedBatchNorm=r.asm.FusedBatchNorm).apply(null,arguments)},r._FusedConv2D=function(){return(r._FusedConv2D=r.asm.FusedConv2D).apply(null,arguments)},r._FusedDepthwiseConv2D=function(){return(r._FusedDepthwiseConv2D=r.asm.FusedDepthwiseConv2D).apply(null,arguments)},r._Gather=function(){return(r._Gather=r.asm.Gather).apply(null,arguments)},r._GatherNd=function(){return(r._GatherNd=r.asm.GatherNd).apply(null,arguments)},r._Greater=function(){return(r._Greater=r.asm.Greater).apply(null,arguments)},r._GreaterEqual=function(){return(r._GreaterEqual=r.asm.GreaterEqual).apply(null,arguments)},r._IsNan=function(){return(r._IsNan=r.asm.IsNan).apply(null,arguments)},r._LeakyRelu=function(){return(r._LeakyRelu=r.asm.LeakyRelu).apply(null,arguments)},r._Less=function(){return(r._Less=r.asm.Less).apply(null,arguments)},r._LessEqual=function(){return(r._LessEqual=r.asm.LessEqual).apply(null,arguments)},r._Log=function(){return(r._Log=r.asm.Log).apply(null,arguments)},r._LogicalAnd=function(){return(r._LogicalAnd=r.asm.LogicalAnd).apply(null,arguments)},r._LogicalNot=function(){return(r._LogicalNot=r.asm.LogicalNot).apply(null,arguments)},r._LogicalOr=function(){return(r._LogicalOr=r.asm.LogicalOr).apply(null,arguments)},r._LogicalXor=function(){return(r._LogicalXor=r.asm.LogicalXor).apply(null,arguments)},r._Max=function(){return(r._Max=r.asm.Max).apply(null,arguments)},r._MaxPool=function(){return(r._MaxPool=r.asm.MaxPool).apply(null,arguments)},r._Maximum=function(){return(r._Maximum=r.asm.Maximum).apply(null,arguments)},r._Mean=function(){return(r._Mean=r.asm.Mean).apply(null,arguments)},r._Min=function(){return(r._Min=r.asm.Min).apply(null,arguments)},r._Minimum=function(){return(r._Minimum=r.asm.Minimum).apply(null,arguments)},r._MirrorPad=function(){return(r._MirrorPad=r.asm.MirrorPad).apply(null,arguments)},r._Multiply=function(){return(r._Multiply=r.asm.Multiply).apply(null,arguments)},r._Neg=function(){return(r._Neg=r.asm.Neg).apply(null,arguments)},r._NonMaxSuppressionV3=function(){return(r._NonMaxSuppressionV3=r.asm.NonMaxSuppressionV3).apply(null,arguments)},r._NonMaxSuppressionV4=function(){return(r._NonMaxSuppressionV4=r.asm.NonMaxSuppressionV4).apply(null,arguments)},r._NonMaxSuppressionV5=function(){return(r._NonMaxSuppressionV5=r.asm.NonMaxSuppressionV5).apply(null,arguments)},r._NotEqual=function(){return(r._NotEqual=r.asm.NotEqual).apply(null,arguments)},r._OneHot=function(){return(r._OneHot=r.asm.OneHot).apply(null,arguments)},r._PadV2=function(){return(r._PadV2=r.asm.PadV2).apply(null,arguments)},r._Pow=function(){return(r._Pow=r.asm.Pow).apply(null,arguments)},r._Prelu=function(){return(r._Prelu=r.asm.Prelu).apply(null,arguments)},r._Prod=function(){return(r._Prod=r.asm.Prod).apply(null,arguments)},r._RealDiv=function(){return(r._RealDiv=r.asm.RealDiv).apply(null,arguments)},r._Reciprocal=function(){return(r._Reciprocal=r.asm.Reciprocal).apply(null,arguments)},r._Relu=function(){return(r._Relu=r.asm.Relu).apply(null,arguments)},r._Relu6=function(){return(r._Relu6=r.asm.Relu6).apply(null,arguments)},r._ResizeBilinear=function(){return(r._ResizeBilinear=r.asm.ResizeBilinear).apply(null,arguments)},r._ResizeNearestNeighbor=function(){return(r._ResizeNearestNeighbor=r.asm.ResizeNearestNeighbor).apply(null,arguments)},r._Reverse=function(){return(r._Reverse=r.asm.Reverse).apply(null,arguments)},r._RotateWithOffset=function(){return(r._RotateWithOffset=r.asm.RotateWithOffset).apply(null,arguments)},r._Round=function(){return(r._Round=r.asm.Round).apply(null,arguments)},r._Rsqrt=function(){return(r._Rsqrt=r.asm.Rsqrt).apply(null,arguments)},r._ScatterNd=function(){return(r._ScatterNd=r.asm.ScatterNd).apply(null,arguments)},r._SelectV2=function(){return(r._SelectV2=r.asm.SelectV2).apply(null,arguments)},r._Sigmoid=function(){return(r._Sigmoid=r.asm.Sigmoid).apply(null,arguments)},r._Sin=function(){return(r._Sin=r.asm.Sin).apply(null,arguments)},r._Softmax=function(){return(r._Softmax=r.asm.Softmax).apply(null,arguments)},r._SparseFillEmptyRows=function(){return(r._SparseFillEmptyRows=r.asm.SparseFillEmptyRows).apply(null,arguments)},r._SparseReshape=function(){return(r._SparseReshape=r.asm.SparseReshape).apply(null,arguments)},r._SparseSegmentReduction=function(){return(r._SparseSegmentReduction=r.asm.SparseSegmentReduction).apply(null,arguments)},r._Sqrt=function(){return(r._Sqrt=r.asm.Sqrt).apply(null,arguments)},r._Square=function(){return(r._Square=r.asm.Square).apply(null,arguments)},r._SquaredDifference=function(){return(r._SquaredDifference=r.asm.SquaredDifference).apply(null,arguments)},r._Step=function(){return(r._Step=r.asm.Step).apply(null,arguments)},r._StridedSlice=function(){return(r._StridedSlice=r.asm.StridedSlice).apply(null,arguments)},r._Sub=function(){return(r._Sub=r.asm.Sub).apply(null,arguments)},r._Sum=function(){return(r._Sum=r.asm.Sum).apply(null,arguments)},r._Tan=function(){return(r._Tan=r.asm.Tan).apply(null,arguments)},r._Tanh=function(){return(r._Tanh=r.asm.Tanh).apply(null,arguments)},r._Tile=function(){return(r._Tile=r.asm.Tile).apply(null,arguments)},r._TopK=function(){return(r._TopK=r.asm.TopK).apply(null,arguments)},r._Transform=function(){return(r._Transform=r.asm.Transform).apply(null,arguments)},r._Transpose=function(){return(r._Transpose=r.asm.Transpose).apply(null,arguments)},r.__FusedMatMul=function(){return(r.__FusedMatMul=r.asm._FusedMatMul).apply(null,arguments)},r._malloc=function(){return(r._malloc=r.asm.malloc).apply(null,arguments)},r._free=function(){return(r._free=r.asm.free).apply(null,arguments)},r.__emscripten_tls_init=function(){return(r.__emscripten_tls_init=r.asm._emscripten_tls_init).apply(null,arguments)};var an=r._pthread_self=function(){return(an=r._pthread_self=r.asm.pthread_self).apply(null,arguments)};r.___errno_location=function(){return(r.___errno_location=r.asm.__errno_location).apply(null,arguments)};var $n=r.__emscripten_thread_init=function(){return($n=r.__emscripten_thread_init=r.asm._emscripten_thread_init).apply(null,arguments)};r.__emscripten_thread_crashed=function(){return(r.__emscripten_thread_crashed=r.asm._emscripten_thread_crashed).apply(null,arguments)},r._emscripten_main_thread_process_queued_calls=function(){return(r._emscripten_main_thread_process_queued_calls=r.asm.emscripten_main_thread_process_queued_calls).apply(null,arguments)},r._emscripten_main_browser_thread_id=function(){return(r._emscripten_main_browser_thread_id=r.asm.emscripten_main_browser_thread_id).apply(null,arguments)};var Gn=r._emscripten_run_in_main_runtime_thread_js=function(){return(Gn=r._emscripten_run_in_main_runtime_thread_js=r.asm.emscripten_run_in_main_runtime_thread_js).apply(null,arguments)};r._emscripten_dispatch_to_thread_=function(){return(r._emscripten_dispatch_to_thread_=r.asm.emscripten_dispatch_to_thread_).apply(null,arguments)};var qn=r.__emscripten_proxy_execute_task_queue=function(){return(qn=r.__emscripten_proxy_execute_task_queue=r.asm._emscripten_proxy_execute_task_queue).apply(null,arguments)},sn=r.__emscripten_thread_free_data=function(){return(sn=r.__emscripten_thread_free_data=r.asm._emscripten_thread_free_data).apply(null,arguments)},Kn=r.__emscripten_thread_exit=function(){return(Kn=r.__emscripten_thread_exit=r.asm._emscripten_thread_exit).apply(null,arguments)},Xn=r._emscripten_stack_set_limits=function(){return(Xn=r._emscripten_stack_set_limits=r.asm.emscripten_stack_set_limits).apply(null,arguments)},Vt=r.stackSave=function(){return(Vt=r.stackSave=r.asm.stackSave).apply(null,arguments)},on=r.stackRestore=function(){return(on=r.stackRestore=r.asm.stackRestore).apply(null,arguments)},un=r.stackAlloc=function(){return(un=r.stackAlloc=r.asm.stackAlloc).apply(null,arguments)};r.dynCall_iijjiiii=function(){return(r.dynCall_iijjiiii=r.asm.dynCall_iijjiiii).apply(null,arguments)},r.dynCall_jiji=function(){return(r.dynCall_jiji=r.asm.dynCall_jiji).apply(null,arguments)},r.keepRuntimeAlive=it,r.wasmMemory=ae,r.cwrap=In,r.ExitStatus=Je,r.PThread=W;var ln;lt=function w(){ln||Ut(),ln||(lt=w)};function Ut(w){if(Ye>0)return;if(I){f(r),ot(),startWorker(r);return}if(Se(),Ye>0)return;function T(){ln||(ln=!0,r.calledRun=!0,!Te&&(ot(),f(r),r.onRuntimeInitialized&&r.onRuntimeInitialized(),Dt()))}r.setStatus?(r.setStatus("Running..."),setTimeout(function(){setTimeout(function(){r.setStatus("")},1),T()},1)):T()}if(r.preInit)for(typeof r.preInit=="function"&&(r.preInit=[r.preInit]);r.preInit.length>0;)r.preInit.pop()();Ut();var cn;g&&(cn={uncaughtException:process.listeners("uncaughtException").filter(function(w){return!g.uncaughtException.indexOf(w)>-1}),unhandledRejection:process.listeners("unhandledRejection").filter(function(w){return!g.unhandledRejection.indexOf(w)>-1})});var pn;if(typeof WasmBackendModule<"u")pn=WasmBackendModule;else if(typeof i<"u")pn=i;else throw new Error("Could not find wasm module in post.js");if(cn){var fr=pn._dispose;pn._dispose=function(){fr(),cn.uncaughtException.forEach(function(w){process.removeListener("uncaughtException",w)}),cn.unhandledRejection.forEach(function(w){process.removeListener("unhandledRejection",w)})}}return i.ready}})();s.exports=a})(Bh);const ji=er,Lh=Ps({__proto__:null,default:ji},[er]);var jh=`"use strict";var Module={};var ENVIRONMENT_IS_NODE=typeof process=="object"&&typeof process.versions=="object"&&typeof process.versions.node=="string";if(ENVIRONMENT_IS_NODE){var nodeWorkerThreads=require("worker_threads");var parentPort=nodeWorkerThreads.parentPort;parentPort.on("message",data=>onmessage({data:data}));var fs=require("fs");Object.assign(global,{self:global,require:require,Module:Module,location:{href:__filename},Worker:nodeWorkerThreads.Worker,importScripts:function(f){(0,eval)(fs.readFileSync(f,"utf8")+"//# sourceURL="+f)},postMessage:function(msg){parentPort.postMessage(msg)},performance:global.performance||{now:function(){return Date.now()}}})}var initializedJS=false;var pendingNotifiedProxyingQueues=[];function threadPrintErr(){var text=Array.prototype.slice.call(arguments).join(" ");if(ENVIRONMENT_IS_NODE){fs.writeSync(2,text+"
");return}console.error(text)}function threadAlert(){var text=Array.prototype.slice.call(arguments).join(" ");postMessage({cmd:"alert",text:text,threadId:Module["_pthread_self"]()})}var err=threadPrintErr;self.alert=threadAlert;Module["instantiateWasm"]=(info,receiveInstance)=>{var instance=new WebAssembly.Instance(Module["wasmModule"],info);receiveInstance(instance);Module["wasmModule"]=null;return instance.exports};self.onunhandledrejection=e=>{throw e.reason??e};self.startWorker=instance=>{Module=instance;postMessage({"cmd":"loaded"})};self.onmessage=e=>{try{if(e.data.cmd==="load"){Module["wasmModule"]=e.data.wasmModule;for(const handler of e.data.handlers){Module[handler]=function(){postMessage({cmd:"callHandler",handler:handler,args:[...arguments]})}}Module["wasmMemory"]=e.data.wasmMemory;Module["buffer"]=Module["wasmMemory"].buffer;Module["ENVIRONMENT_IS_PTHREAD"]=true;if(typeof e.data.urlOrBlob=="string"){importScripts(e.data.urlOrBlob)}else{var objectUrl=URL.createObjectURL(e.data.urlOrBlob);importScripts(objectUrl);URL.revokeObjectURL(objectUrl)}WasmBackendModuleThreadedSimd(Module)}else if(e.data.cmd==="run"){Module["__emscripten_thread_init"](e.data.pthread_ptr,0,0,1);Module["establishStackSpace"]();Module["PThread"].receiveObjectTransfer(e.data);Module["PThread"].threadInitTLS();if(!initializedJS){pendingNotifiedProxyingQueues.forEach(queue=>{Module["executeNotifiedProxyingQueue"](queue)});pendingNotifiedProxyingQueues=[];initializedJS=true}try{Module["invokeEntryPoint"](e.data.start_routine,e.data.arg)}catch(ex){if(ex!="unwind"){if(ex instanceof Module["ExitStatus"]){if(Module["keepRuntimeAlive"]()){}else{Module["__emscripten_thread_exit"](ex.status)}}else{throw ex}}}}else if(e.data.cmd==="cancel"){if(Module["_pthread_self"]()){Module["__emscripten_thread_exit"](-1)}}else if(e.data.target==="setimmediate"){}else if(e.data.cmd==="processProxyingQueue"){if(initializedJS){Module["executeNotifiedProxyingQueue"](e.data.queue)}else{pendingNotifiedProxyingQueues.push(e.data.queue)}}else if(e.data.cmd){err("worker.js received unknown command "+e.data.cmd);err(e.data)}}catch(ex){if(Module["__emscripten_thread_crashed"]){Module["__emscripten_thread_crashed"]()}throw ex}};`,tr={},Wh={get exports(){return tr},set exports(s){tr=s}};(function(s,e){var a=(()=>{var o=typeof document<"u"&&document.currentScript?document.currentScript.src:void 0;return typeof __filename<"u"&&(o=o||__filename),function(i){i=i||{};var t=typeof i<"u"?i:{},u,c;t.ready=new Promise(function(F,L){u=F,c=L});var p;typeof process<"u"&&process.listeners&&(p={uncaughtException:process.listeners("uncaughtException"),unhandledRejection:process.listeners("unhandledRejection")});var h=Object.assign({},t),r=typeof window=="object",f=typeof importScripts=="function",b=typeof process=="object"&&typeof process.versions=="object"&&typeof process.versions.node=="string",g="";function _(F){return t.locateFile?t.locateFile(F,g):g+F}var v,x,S;if(b){var k=zt,I=zt;f?g=I.dirname(g)+"/":g=__dirname+"/",v=(F,L)=>(F=st(F)?new URL(F):I.normalize(F),k.readFileSync(F,L?void 0:"utf8")),S=F=>{var L=v(F,!0);return L.buffer||(L=new Uint8Array(L)),L},x=(F,L,U)=>{F=st(F)?new URL(F):I.normalize(F),k.readFile(F,function(W,se){W?U(W):L(se.buffer)})},process.argv.length>1&&process.argv[1].replace(/\\/g,"/"),process.argv.slice(2),process.on("uncaughtException",function(F){if(!(F instanceof it))throw F}),process.on("unhandledRejection",function(F){throw F}),t.inspect=function(){return"[Emscripten Module object]"}}else(r||f)&&(f?g=self.location.href:typeof document<"u"&&document.currentScript&&(g=document.currentScript.src),o&&(g=o),g.indexOf("blob:")!==0?g=g.substr(0,g.replace(/[?#].*/,"").lastIndexOf("/")+1):g="",v=F=>{var L=new XMLHttpRequest;return L.open("GET",F,!1),L.send(null),L.responseText},f&&(S=F=>{var L=new XMLHttpRequest;return L.open("GET",F,!1),L.responseType="arraybuffer",L.send(null),new Uint8Array(L.response)}),x=(F,L,U)=>{var W=new XMLHttpRequest;W.open("GET",F,!0),W.responseType="arraybuffer",W.onload=()=>{if(W.status==200||W.status==0&&W.response){L(W.response);return}U()},W.onerror=U,W.send(null)});var R=t.print||console.log.bind(console),N=t.printErr||console.warn.bind(console);Object.assign(t,h),h=null,t.arguments&&t.arguments,t.thisProgram&&t.thisProgram,t.quit&&t.quit;var E;t.wasmBinary&&(E=t.wasmBinary),t.noExitRuntime,typeof WebAssembly!="object"&&Ke("no native wasm support detected");var O,j=!1,B=typeof TextDecoder<"u"?new TextDecoder("utf8"):void 0;function D(F,L,U){for(var W=L+U,se=L;F[se]&&!(se>=W);)++se;if(se-L>16&&F.buffer&&B)return B.decode(F.subarray(L,se));for(var J="";L<se;){var te=F[L++];if(!(te&128)){J+=String.fromCharCode(te);continue}var Z=F[L++]&63;if((te&224)==192){J+=String.fromCharCode((te&31)<<6|Z);continue}var pe=F[L++]&63;if((te&240)==224?te=(te&15)<<12|Z<<6|pe:te=(te&7)<<18|Z<<12|pe<<6|F[L++]&63,te<65536)J+=String.fromCharCode(te);else{var Ce=te-65536;J+=String.fromCharCode(55296|Ce>>10,56320|Ce&1023)}}return J}function V(F,L){return F?D(ee,F,L):""}function q(F,L,U,W){if(!(W>0))return 0;for(var se=U,J=U+W-1,te=0;te<F.length;++te){var Z=F.charCodeAt(te);if(Z>=55296&&Z<=57343){var pe=F.charCodeAt(++te);Z=65536+((Z&1023)<<10)|pe&1023}if(Z<=127){if(U>=J)break;L[U++]=Z}else if(Z<=2047){if(U+1>=J)break;L[U++]=192|Z>>6,L[U++]=128|Z&63}else if(Z<=65535){if(U+2>=J)break;L[U++]=224|Z>>12,L[U++]=128|Z>>6&63,L[U++]=128|Z&63}else{if(U+3>=J)break;L[U++]=240|Z>>18,L[U++]=128|Z>>12&63,L[U++]=128|Z>>6&63,L[U++]=128|Z&63}}return L[U]=0,U-se}function $(F,L,U){return q(F,ee,L,U)}var Y,Q,ee,ce;function ae(F){Y=F,t.HEAP8=Q=new Int8Array(F),t.HEAP16=new Int16Array(F),t.HEAP32=new Int32Array(F),t.HEAPU8=ee=new Uint8Array(F),t.HEAPU16=new Uint16Array(F),t.HEAPU32=ce=new Uint32Array(F),t.HEAPF32=new Float32Array(F),t.HEAPF64=new Float64Array(F)}t.INITIAL_MEMORY;var xe=[],Te=[],Ee=[];function qe(){if(t.preRun)for(typeof t.preRun=="function"&&(t.preRun=[t.preRun]);t.preRun.length;)bt(t.preRun.shift());Se(xe)}function ke(){Se(Te)}function yt(){if(t.postRun)for(typeof t.postRun=="function"&&(t.postRun=[t.postRun]);t.postRun.length;)It(t.postRun.shift());Se(Ee)}function bt(F){xe.unshift(F)}function nt(F){Te.unshift(F)}function It(F){Ee.unshift(F)}var ge=0,at=null;function Nt(F){ge++,t.monitorRunDependencies&&t.monitorRunDependencies(ge)}function xt(F){if(ge--,t.monitorRunDependencies&&t.monitorRunDependencies(ge),ge==0&&at){var L=at;at=null,L()}}function Ke(F){t.onAbort&&t.onAbort(F),F="Aborted("+F+")",N(F),j=!0,F+=". Build with -sASSERTIONS for more info.";var L=new WebAssembly.RuntimeError(F);throw c(L),L}var Ot="data:application/octet-stream;base64,";function Xe(F){return F.startsWith(Ot)}function st(F){return F.startsWith("file://")}var he;he="tfjs-backend-wasm.wasm",Xe(he)||(he=_(he));function je(F){try{if(F==he&&E)return new Uint8Array(E);if(S)return S(F);throw"both async and sync fetching of the wasm failed"}catch(L){Ke(L)}}function Zt(){if(!E&&(r||f)){if(typeof fetch=="function"&&!st(he))return fetch(he,{credentials:"same-origin"}).then(function(F){if(!F.ok)throw"failed to load wasm binary file at '"+he+"'";return F.arrayBuffer()}).catch(function(){return je(he)});if(x)return new Promise(function(F,L){x(he,function(U){F(new Uint8Array(U))},L)})}return Promise.resolve().then(function(){return je(he)})}function Pt(){var F={env:St,wasi_snapshot_preview1:St};function L(te,Z){var pe=te.exports;t.asm=pe,O=t.asm.memory,ae(O.buffer),t.asm.__indirect_function_table,nt(t.asm.__wasm_call_ctors),xt()}Nt();function U(te){L(te.instance)}function W(te){return Zt().then(function(Z){return WebAssembly.instantiate(Z,F)}).then(function(Z){return Z}).then(te,function(Z){N("failed to asynchronously prepare wasm: "+Z),Ke(Z)})}function se(){return!E&&typeof WebAssembly.instantiateStreaming=="function"&&!Xe(he)&&!st(he)&&!b&&typeof fetch=="function"?fetch(he,{credentials:"same-origin"}).then(function(te){var Z=WebAssembly.instantiateStreaming(te,F);return Z.then(U,function(pe){return N("wasm streaming compile failed: "+pe),N("falling back to ArrayBuffer instantiation"),W(U)})}):W(U)}if(t.instantiateWasm)try{var J=t.instantiateWasm(F,L);return J}catch(te){N("Module.instantiateWasm callback failed with error: "+te),c(te)}return se().catch(c),{}}function it(F){this.name="ExitStatus",this.message="Program terminated with exit("+F+")",this.status=F}function Se(F){for(;F.length>0;)F.shift()(t)}function ot(){Ke("")}function Dt(){return 2147483648}function Bt(){return Dt()}function We(F,L,U){ee.copyWithin(F,L,L+U)}function ut(F){try{return O.grow(F-Y.byteLength+65535>>>16),ae(O.buffer),1}catch{}}function Ye(F){var L=ee.length;F=F>>>0;var U=Dt();if(F>U)return!1;let W=(pe,Ce)=>pe+(Ce-pe%Ce)%Ce;for(var se=1;se<=4;se*=2){var J=L*(1+.2/se);J=Math.min(J,F+100663296);var te=Math.min(U,W(Math.max(F,J),65536)),Z=ut(te);if(Z)return!0}return!1}function lt(F){return 52}function _t(F,L,U,W,se){return 70}var en=[null,[],[]];function Qe(F,L){var U=en[F];L===0||L===10?((F===1?R:N)(D(U,0)),U.length=0):U.push(L)}function bn(F,L,U,W){for(var se=0,J=0;J<U;J++){var te=ce[L>>2],Z=ce[L+4>>2];L+=8;for(var pe=0;pe<Z;pe++)Qe(F,ee[te+pe]);se+=Z}return ce[W>>2]=se,0}function ct(F){var L=t["_"+F];return L}function vt(F,L){Q.set(F,L)}function _e(F,L,U,W,se){var J={string:ve=>{var et=0;if(ve!=null&&ve!==0){var Ne=(ve.length<<2)+1;et=Je(Ne),$(ve,et,Ne)}return et},array:ve=>{var et=Je(ve.length);return vt(ve,et),et}};function te(ve){return L==="string"?V(ve):L==="boolean"?Boolean(ve):ve}var Z=ct(F),pe=[],Ce=0;if(W)for(var Ze=0;Ze<W.length;Ze++){var At=J[U[Ze]];At?(Ce===0&&(Ce=pt()),pe[Ze]=At(W[Ze])):pe[Ze]=W[Ze]}var Ht=Z.apply(null,pe);function ft(ve){return Ce!==0&&jt(Ce),te(ve)}return Ht=ft(Ht),Ht}function Lt(F,L,U,W){U=U||[];var se=U.every(te=>te==="number"||te==="boolean"),J=L!=="string";return J&&se&&!W?ct(F):function(){return _e(F,L,U,arguments)}}var St={abort:ot,emscripten_get_heap_max:Bt,emscripten_memcpy_big:We,emscripten_resize_heap:Ye,fd_close:lt,fd_seek:_t,fd_write:bn};Pt(),t.___wasm_call_ctors=function(){return(t.___wasm_call_ctors=t.asm.__wasm_call_ctors).apply(null,arguments)},t._init=function(){return(t._init=t.asm.init).apply(null,arguments)},t._init_with_threads_count=function(){return(t._init_with_threads_count=t.asm.init_with_threads_count).apply(null,arguments)},t._get_threads_count=function(){return(t._get_threads_count=t.asm.get_threads_count).apply(null,arguments)},t._register_tensor=function(){return(t._register_tensor=t.asm.register_tensor).apply(null,arguments)},t._dispose_data=function(){return(t._dispose_data=t.asm.dispose_data).apply(null,arguments)},t._dispose=function(){return(t._dispose=t.asm.dispose).apply(null,arguments)},t._Abs=function(){return(t._Abs=t.asm.Abs).apply(null,arguments)},t._Add=function(){return(t._Add=t.asm.Add).apply(null,arguments)},t._AddN=function(){return(t._AddN=t.asm.AddN).apply(null,arguments)},t._All=function(){return(t._All=t.asm.All).apply(null,arguments)},t._Any=function(){return(t._Any=t.asm.Any).apply(null,arguments)},t._ArgMax=function(){return(t._ArgMax=t.asm.ArgMax).apply(null,arguments)},t._AvgPool=function(){return(t._AvgPool=t.asm.AvgPool).apply(null,arguments)},t._BatchMatMul=function(){return(t._BatchMatMul=t.asm.BatchMatMul).apply(null,arguments)},t._Ceil=function(){return(t._Ceil=t.asm.Ceil).apply(null,arguments)},t._ClipByValue=function(){return(t._ClipByValue=t.asm.ClipByValue).apply(null,arguments)},t._Conv2D=function(){return(t._Conv2D=t.asm.Conv2D).apply(null,arguments)},t._Conv2DBackpropInput=function(){return(t._Conv2DBackpropInput=t.asm.Conv2DBackpropInput).apply(null,arguments)},t._Cos=function(){return(t._Cos=t.asm.Cos).apply(null,arguments)},t._Cosh=function(){return(t._Cosh=t.asm.Cosh).apply(null,arguments)},t._CropAndResize=function(){return(t._CropAndResize=t.asm.CropAndResize).apply(null,arguments)},t._Cumprod=function(){return(t._Cumprod=t.asm.Cumprod).apply(null,arguments)},t._Cumsum=function(){return(t._Cumsum=t.asm.Cumsum).apply(null,arguments)},t._DepthToSpace=function(){return(t._DepthToSpace=t.asm.DepthToSpace).apply(null,arguments)},t._DepthwiseConv2dNative=function(){return(t._DepthwiseConv2dNative=t.asm.DepthwiseConv2dNative).apply(null,arguments)},t._Elu=function(){return(t._Elu=t.asm.Elu).apply(null,arguments)},t._Equal=function(){return(t._Equal=t.asm.Equal).apply(null,arguments)},t._Exp=function(){return(t._Exp=t.asm.Exp).apply(null,arguments)},t._FlipLeftRight=function(){return(t._FlipLeftRight=t.asm.FlipLeftRight).apply(null,arguments)},t._Floor=function(){return(t._Floor=t.asm.Floor).apply(null,arguments)},t._FloorDiv=function(){return(t._FloorDiv=t.asm.FloorDiv).apply(null,arguments)},t._FusedBatchNorm=function(){return(t._FusedBatchNorm=t.asm.FusedBatchNorm).apply(null,arguments)},t._FusedConv2D=function(){return(t._FusedConv2D=t.asm.FusedConv2D).apply(null,arguments)},t._FusedDepthwiseConv2D=function(){return(t._FusedDepthwiseConv2D=t.asm.FusedDepthwiseConv2D).apply(null,arguments)},t._Gather=function(){return(t._Gather=t.asm.Gather).apply(null,arguments)},t._GatherNd=function(){return(t._GatherNd=t.asm.GatherNd).apply(null,arguments)},t._Greater=function(){return(t._Greater=t.asm.Greater).apply(null,arguments)},t._GreaterEqual=function(){return(t._GreaterEqual=t.asm.GreaterEqual).apply(null,arguments)},t._IsNan=function(){return(t._IsNan=t.asm.IsNan).apply(null,arguments)},t._LeakyRelu=function(){return(t._LeakyRelu=t.asm.LeakyRelu).apply(null,arguments)},t._Less=function(){return(t._Less=t.asm.Less).apply(null,arguments)},t._LessEqual=function(){return(t._LessEqual=t.asm.LessEqual).apply(null,arguments)},t._Log=function(){return(t._Log=t.asm.Log).apply(null,arguments)},t._LogicalAnd=function(){return(t._LogicalAnd=t.asm.LogicalAnd).apply(null,arguments)},t._LogicalNot=function(){return(t._LogicalNot=t.asm.LogicalNot).apply(null,arguments)},t._LogicalOr=function(){return(t._LogicalOr=t.asm.LogicalOr).apply(null,arguments)},t._LogicalXor=function(){return(t._LogicalXor=t.asm.LogicalXor).apply(null,arguments)},t._Max=function(){return(t._Max=t.asm.Max).apply(null,arguments)},t._MaxPool=function(){return(t._MaxPool=t.asm.MaxPool).apply(null,arguments)},t._Maximum=function(){return(t._Maximum=t.asm.Maximum).apply(null,arguments)},t._Mean=function(){return(t._Mean=t.asm.Mean).apply(null,arguments)},t._Min=function(){return(t._Min=t.asm.Min).apply(null,arguments)},t._Minimum=function(){return(t._Minimum=t.asm.Minimum).apply(null,arguments)},t._MirrorPad=function(){return(t._MirrorPad=t.asm.MirrorPad).apply(null,arguments)},t._Multiply=function(){return(t._Multiply=t.asm.Multiply).apply(null,arguments)},t._Neg=function(){return(t._Neg=t.asm.Neg).apply(null,arguments)},t._NonMaxSuppressionV3=function(){return(t._NonMaxSuppressionV3=t.asm.NonMaxSuppressionV3).apply(null,arguments)},t._NonMaxSuppressionV4=function(){return(t._NonMaxSuppressionV4=t.asm.NonMaxSuppressionV4).apply(null,arguments)},t._NonMaxSuppressionV5=function(){return(t._NonMaxSuppressionV5=t.asm.NonMaxSuppressionV5).apply(null,arguments)},t._NotEqual=function(){return(t._NotEqual=t.asm.NotEqual).apply(null,arguments)},t._OneHot=function(){return(t._OneHot=t.asm.OneHot).apply(null,arguments)},t._PadV2=function(){return(t._PadV2=t.asm.PadV2).apply(null,arguments)},t._Pow=function(){return(t._Pow=t.asm.Pow).apply(null,arguments)},t._Prelu=function(){return(t._Prelu=t.asm.Prelu).apply(null,arguments)},t._Prod=function(){return(t._Prod=t.asm.Prod).apply(null,arguments)},t._RealDiv=function(){return(t._RealDiv=t.asm.RealDiv).apply(null,arguments)},t._Reciprocal=function(){return(t._Reciprocal=t.asm.Reciprocal).apply(null,arguments)},t._Relu=function(){return(t._Relu=t.asm.Relu).apply(null,arguments)},t._Relu6=function(){return(t._Relu6=t.asm.Relu6).apply(null,arguments)},t._ResizeBilinear=function(){return(t._ResizeBilinear=t.asm.ResizeBilinear).apply(null,arguments)},t._ResizeNearestNeighbor=function(){return(t._ResizeNearestNeighbor=t.asm.ResizeNearestNeighbor).apply(null,arguments)},t._Reverse=function(){return(t._Reverse=t.asm.Reverse).apply(null,arguments)},t._RotateWithOffset=function(){return(t._RotateWithOffset=t.asm.RotateWithOffset).apply(null,arguments)},t._Round=function(){return(t._Round=t.asm.Round).apply(null,arguments)},t._Rsqrt=function(){return(t._Rsqrt=t.asm.Rsqrt).apply(null,arguments)},t._ScatterNd=function(){return(t._ScatterNd=t.asm.ScatterNd).apply(null,arguments)},t._SelectV2=function(){return(t._SelectV2=t.asm.SelectV2).apply(null,arguments)},t._Sigmoid=function(){return(t._Sigmoid=t.asm.Sigmoid).apply(null,arguments)},t._Sin=function(){return(t._Sin=t.asm.Sin).apply(null,arguments)},t._Softmax=function(){return(t._Softmax=t.asm.Softmax).apply(null,arguments)},t._SparseFillEmptyRows=function(){return(t._SparseFillEmptyRows=t.asm.SparseFillEmptyRows).apply(null,arguments)},t._SparseReshape=function(){return(t._SparseReshape=t.asm.SparseReshape).apply(null,arguments)},t._SparseSegmentReduction=function(){return(t._SparseSegmentReduction=t.asm.SparseSegmentReduction).apply(null,arguments)},t._Sqrt=function(){return(t._Sqrt=t.asm.Sqrt).apply(null,arguments)},t._Square=function(){return(t._Square=t.asm.Square).apply(null,arguments)},t._SquaredDifference=function(){return(t._SquaredDifference=t.asm.SquaredDifference).apply(null,arguments)},t._Step=function(){return(t._Step=t.asm.Step).apply(null,arguments)},t._StridedSlice=function(){return(t._StridedSlice=t.asm.StridedSlice).apply(null,arguments)},t._Sub=function(){return(t._Sub=t.asm.Sub).apply(null,arguments)},t._Sum=function(){return(t._Sum=t.asm.Sum).apply(null,arguments)},t._Tan=function(){return(t._Tan=t.asm.Tan).apply(null,arguments)},t._Tanh=function(){return(t._Tanh=t.asm.Tanh).apply(null,arguments)},t._Tile=function(){return(t._Tile=t.asm.Tile).apply(null,arguments)},t._TopK=function(){return(t._TopK=t.asm.TopK).apply(null,arguments)},t._Transform=function(){return(t._Transform=t.asm.Transform).apply(null,arguments)},t._Transpose=function(){return(t._Transpose=t.asm.Transpose).apply(null,arguments)},t.__FusedMatMul=function(){return(t.__FusedMatMul=t.asm._FusedMatMul).apply(null,arguments)},t._malloc=function(){return(t._malloc=t.asm.malloc).apply(null,arguments)},t._free=function(){return(t._free=t.asm.free).apply(null,arguments)},t.___errno_location=function(){return(t.___errno_location=t.asm.__errno_location).apply(null,arguments)};var pt=t.stackSave=function(){return(pt=t.stackSave=t.asm.stackSave).apply(null,arguments)},jt=t.stackRestore=function(){return(jt=t.stackRestore=t.asm.stackRestore).apply(null,arguments)},Je=t.stackAlloc=function(){return(Je=t.stackAlloc=t.asm.stackAlloc).apply(null,arguments)};t.dynCall_iijjiiii=function(){return(t.dynCall_iijjiiii=t.asm.dynCall_iijjiiii).apply(null,arguments)},t.dynCall_jiji=function(){return(t.dynCall_jiji=t.asm.dynCall_jiji).apply(null,arguments)},t.cwrap=Lt;var Mt;at=function F(){Mt||Wt(),Mt||(at=F)};function Wt(F){if(ge>0||(qe(),ge>0))return;function L(){Mt||(Mt=!0,t.calledRun=!0,!j&&(ke(),u(t),t.onRuntimeInitialized&&t.onRuntimeInitialized(),yt()))}t.setStatus?(t.setStatus("Running..."),setTimeout(function(){setTimeout(function(){t.setStatus("")},1),L()},1)):L()}if(t.preInit)for(typeof t.preInit=="function"&&(t.preInit=[t.preInit]);t.preInit.length>0;)t.preInit.pop()();Wt();var dt;p&&(dt={uncaughtException:process.listeners("uncaughtException").filter(function(F){return!p.uncaughtException.indexOf(F)>-1}),unhandledRejection:process.listeners("unhandledRejection").filter(function(F){return!p.unhandledRejection.indexOf(F)>-1})});var ht;if(typeof i<"u")ht=i;else if(typeof WasmBackendModuleThreadedSimd<"u")ht=WasmBackendModuleThreadedSimd;else throw new Error("Could not find wasm module in post.js");if(dt){var Re=ht._dispose;ht._dispose=function(){Re(),dt.uncaughtException.forEach(function(F){process.removeListener("uncaughtException",F)}),dt.unhandledRejection.forEach(function(F){process.removeListener("unhandledRejection",F)})}}return i.ready}})();s.exports=a})(Wh);const Wi=tr,Hh=Ps({__proto__:null,default:Wi},[tr]);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ea=ji||Lh,Vh=Wi||Hh;class Uh extends Vu{constructor(e){super(),this.wasm=e,this.dataIdNextNumber=1,this.wasm.tfjs.initWithThreadsCount(Kh),this.wasm.tfjs.getThreadsCount(),this.dataIdMap=new Uu(this,Ns())}write(e,a,o){const i={id:this.dataIdNextNumber++};return this.move(i,e,a,o,1),i}numDataIds(){return this.dataIdMap.numDataIds()}async time(e){const a=Kr();return e(),{kernelMs:Kr()-a}}move(e,a,o,i,t){const u=this.dataIdNextNumber++;if(i==="string"){const r=a;this.dataIdMap.set(e,{id:u,stringBytes:r,shape:o,dtype:i,memoryOffset:null,refCount:t});return}const c=X(o),p=c*Xr(i),h=this.wasm._malloc(p);this.dataIdMap.set(e,{id:u,memoryOffset:h,shape:o,dtype:i,refCount:t}),this.wasm.tfjs.registerTensor(u,c,h),a!=null&&this.wasm.HEAPU8.set(new Uint8Array(a.buffer,a.byteOffset,p),h)}async read(e){return this.readSync(e)}readSync(e,a,o){const{memoryOffset:i,dtype:t,shape:u,stringBytes:c}=this.dataIdMap.get(e);if(t==="string")return(a==null||a===0)&&(o==null||o>=c.length)?c:c.slice(a,o);a=a||0,o=o||X(u);const p=Xr(t),h=this.wasm.HEAPU8.slice(i+a*p,i+o*p);return Gh(h.buffer,t)}disposeData(e,a=!1){if(this.dataIdMap.has(e)){const o=this.dataIdMap.get(e);if(o.refCount--,!a&&o.refCount>0)return!1;this.wasm._free(o.memoryOffset),this.wasm.tfjs.disposeData(o.id),this.dataIdMap.delete(e)}return!0}refCount(e){return this.dataIdMap.has(e)?this.dataIdMap.get(e).refCount:0}incRef(e){const a=this.dataIdMap.get(e);a!=null&&a.refCount++}floatPrecision(){return 32}getMemoryOffset(e){return this.dataIdMap.get(e).memoryOffset}dispose(){this.wasm.tfjs.dispose(),"PThread"in this.wasm&&this.wasm.PThread.terminateAllThreads(),this.wasm=null}memory(){return{unreliable:!1}}makeOutput(e,a,o){let i;if(o==null)i=this.write(null,e,a);else{const t=this.dataIdNextNumber++;i={id:t},this.dataIdMap.set(i,{id:t,memoryOffset:o,shape:e,dtype:a,refCount:1});const u=X(e);this.wasm.tfjs.registerTensor(t,u,o)}return{dataId:i,shape:e,dtype:a}}typedArrayFromHeap({shape:e,dtype:a,dataId:o}){const i=this.wasm.HEAPU8.buffer,{memoryOffset:t}=this.dataIdMap.get(o),u=X(e);switch(a){case"float32":return new Float32Array(i,t,u);case"int32":return new Int32Array(i,t,u);case"bool":return new Uint8Array(i,t,u);default:throw new Error(`Unknown dtype ${a}`)}}}function zh(s,e,a){let o="tfjs-backend-wasm.wasm";return s&&e?o="tfjs-backend-wasm-threaded-simd.wasm":s&&(o="tfjs-backend-wasm-simd.wasm"),br!=null&&br[o]!=null?br[o]:a+o}async function $h(){const[s,e]=await Promise.all([Jn().getAsync("WASM_HAS_SIMD_SUPPORT"),Jn().getAsync("WASM_HAS_MULTITHREAD_SUPPORT")]);return new Promise((a,o)=>{const i={};i.locateFile=(c,p)=>{if(c.endsWith(".worker.js")){const h=jh.replace(/\n/g,"\\n"),r=new Blob([h],{type:"application/javascript"});return URL.createObjectURL(r)}return c.endsWith(".wasm")?zh(s,e,p):p+c};let t=!1;i.onAbort=()=>{if(t||_r)return;_r=!0,o({message:"Make sure the server can serve the `.wasm` file relative to the bundled js file. For more details see https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers"})};let u;e&&s&&qh==null?(i.mainScriptUrlOrBlob=new Blob(["var WasmBackendModuleThreadedSimd = "+ea.toString()],{type:"text/javascript"}),u=ea(i)):u=Vh(i),u.then(c=>{t=!0,_r=!1;const p=null;c.tfjs={init:c.cwrap("init",null,[]),initWithThreadsCount:c.cwrap("init_with_threads_count",null,["number"]),getThreadsCount:c.cwrap("get_threads_count","number",[]),registerTensor:c.cwrap("register_tensor",null,["number","number","number"]),disposeData:c.cwrap("dispose_data",p,["number"]),dispose:c.cwrap("dispose",p,[])},a({wasm:c})}).catch(o)})}function Gh(s,e){switch(e){case"float32":return new Float32Array(s);case"int32":return new Int32Array(s);case"bool":return new Uint8Array(s);default:throw new Error(`Unknown dtype ${e}`)}}let qh=null,br={},_r=!1,Kh=-1;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xh=2;zu("wasm",async()=>{const{wasm:s}=await $h();return new Uh(s)},Xh);var Yh={};(function(){var s;function e(n){var l=0;return function(){return l<n.length?{done:!1,value:n[l++]}:{done:!0}}}var a=typeof Object.defineProperties=="function"?Object.defineProperty:function(n,l,d){return n==Array.prototype||n==Object.prototype||(n[l]=d.value),n};function o(n){n=[typeof globalThis=="object"&&globalThis,n,typeof window=="object"&&window,typeof self=="object"&&self,typeof Tn=="object"&&Tn];for(var l=0;l<n.length;++l){var d=n[l];if(d&&d.Math==Math)return d}throw Error("Cannot find global object")}var i=o(this);function t(n,l){if(l)e:{var d=i;n=n.split(".");for(var m=0;m<n.length-1;m++){var y=n[m];if(!(y in d))break e;d=d[y]}n=n[n.length-1],m=d[n],l=l(m),l!=m&&l!=null&&a(d,n,{configurable:!0,writable:!0,value:l})}}t("Symbol",function(n){function l(M){if(this instanceof l)throw new TypeError("Symbol is not a constructor");return new d(m+(M||"")+"_"+y++,M)}function d(M,A){this.g=M,a(this,"description",{configurable:!0,writable:!0,value:A})}if(n)return n;d.prototype.toString=function(){return this.g};var m="jscomp_symbol_"+(1e9*Math.random()>>>0)+"_",y=0;return l}),t("Symbol.iterator",function(n){if(n)return n;n=Symbol("Symbol.iterator");for(var l="Array Int8Array Uint8Array Uint8ClampedArray Int16Array Uint16Array Int32Array Uint32Array Float32Array Float64Array".split(" "),d=0;d<l.length;d++){var m=i[l[d]];typeof m=="function"&&typeof m.prototype[n]!="function"&&a(m.prototype,n,{configurable:!0,writable:!0,value:function(){return u(e(this))}})}return n});function u(n){return n={next:n},n[Symbol.iterator]=function(){return this},n}function c(n){var l=typeof Symbol<"u"&&Symbol.iterator&&n[Symbol.iterator];return l?l.call(n):{next:e(n)}}function p(n){if(!(n instanceof Array)){n=c(n);for(var l,d=[];!(l=n.next()).done;)d.push(l.value);n=d}return n}var h=typeof Object.create=="function"?Object.create:function(n){function l(){}return l.prototype=n,new l},r;if(typeof Object.setPrototypeOf=="function")r=Object.setPrototypeOf;else{var f;e:{var b={a:!0},g={};try{g.__proto__=b,f=g.a;break e}catch{}f=!1}r=f?function(n,l){if(n.__proto__=l,n.__proto__!==l)throw new TypeError(n+" is not extensible");return n}:null}var _=r;function v(n,l){if(n.prototype=h(l.prototype),n.prototype.constructor=n,_)_(n,l);else for(var d in l)if(d!="prototype")if(Object.defineProperties){var m=Object.getOwnPropertyDescriptor(l,d);m&&Object.defineProperty(n,d,m)}else n[d]=l[d];n.ea=l.prototype}function x(){this.l=!1,this.i=null,this.h=void 0,this.g=1,this.s=this.m=0,this.j=null}function S(n){if(n.l)throw new TypeError("Generator is already running");n.l=!0}x.prototype.o=function(n){this.h=n};function k(n,l){n.j={U:l,V:!0},n.g=n.m||n.s}x.prototype.return=function(n){this.j={return:n},this.g=this.s};function I(n,l,d){return n.g=d,{value:l}}function R(n){this.g=new x,this.h=n}function N(n,l){S(n.g);var d=n.g.i;return d?E(n,"return"in d?d.return:function(m){return{value:m,done:!0}},l,n.g.return):(n.g.return(l),O(n))}function E(n,l,d,m){try{var y=l.call(n.g.i,d);if(!(y instanceof Object))throw new TypeError("Iterator result "+y+" is not an object");if(!y.done)return n.g.l=!1,y;var M=y.value}catch(A){return n.g.i=null,k(n.g,A),O(n)}return n.g.i=null,m.call(n.g,M),O(n)}function O(n){for(;n.g.g;)try{var l=n.h(n.g);if(l)return n.g.l=!1,{value:l.value,done:!1}}catch(d){n.g.h=void 0,k(n.g,d)}if(n.g.l=!1,n.g.j){if(l=n.g.j,n.g.j=null,l.V)throw l.U;return{value:l.return,done:!0}}return{value:void 0,done:!0}}function j(n){this.next=function(l){return S(n.g),n.g.i?l=E(n,n.g.i.next,l,n.g.o):(n.g.o(l),l=O(n)),l},this.throw=function(l){return S(n.g),n.g.i?l=E(n,n.g.i.throw,l,n.g.o):(k(n.g,l),l=O(n)),l},this.return=function(l){return N(n,l)},this[Symbol.iterator]=function(){return this}}function B(n){function l(m){return n.next(m)}function d(m){return n.throw(m)}return new Promise(function(m,y){function M(A){A.done?m(A.value):Promise.resolve(A.value).then(l,d).then(M,y)}M(n.next())})}function D(n){return B(new j(new R(n)))}t("Promise",function(n){function l(A){this.h=0,this.i=void 0,this.g=[],this.o=!1;var P=this.j();try{A(P.resolve,P.reject)}catch(C){P.reject(C)}}function d(){this.g=null}function m(A){return A instanceof l?A:new l(function(P){P(A)})}if(n)return n;d.prototype.h=function(A){if(this.g==null){this.g=[];var P=this;this.i(function(){P.l()})}this.g.push(A)};var y=i.setTimeout;d.prototype.i=function(A){y(A,0)},d.prototype.l=function(){for(;this.g&&this.g.length;){var A=this.g;this.g=[];for(var P=0;P<A.length;++P){var C=A[P];A[P]=null;try{C()}catch(H){this.j(H)}}}this.g=null},d.prototype.j=function(A){this.i(function(){throw A})},l.prototype.j=function(){function A(H){return function(K){C||(C=!0,H.call(P,K))}}var P=this,C=!1;return{resolve:A(this.C),reject:A(this.l)}},l.prototype.C=function(A){if(A===this)this.l(new TypeError("A Promise cannot resolve to itself"));else if(A instanceof l)this.F(A);else{e:switch(typeof A){case"object":var P=A!=null;break e;case"function":P=!0;break e;default:P=!1}P?this.v(A):this.m(A)}},l.prototype.v=function(A){var P=void 0;try{P=A.then}catch(C){this.l(C);return}typeof P=="function"?this.G(P,A):this.m(A)},l.prototype.l=function(A){this.s(2,A)},l.prototype.m=function(A){this.s(1,A)},l.prototype.s=function(A,P){if(this.h!=0)throw Error("Cannot settle("+A+", "+P+"): Promise already settled in state"+this.h);this.h=A,this.i=P,this.h===2&&this.D(),this.A()},l.prototype.D=function(){var A=this;y(function(){if(A.B()){var P=i.console;typeof P<"u"&&P.error(A.i)}},1)},l.prototype.B=function(){if(this.o)return!1;var A=i.CustomEvent,P=i.Event,C=i.dispatchEvent;return typeof C>"u"?!0:(typeof A=="function"?A=new A("unhandledrejection",{cancelable:!0}):typeof P=="function"?A=new P("unhandledrejection",{cancelable:!0}):(A=i.document.createEvent("CustomEvent"),A.initCustomEvent("unhandledrejection",!1,!0,A)),A.promise=this,A.reason=this.i,C(A))},l.prototype.A=function(){if(this.g!=null){for(var A=0;A<this.g.length;++A)M.h(this.g[A]);this.g=null}};var M=new d;return l.prototype.F=function(A){var P=this.j();A.J(P.resolve,P.reject)},l.prototype.G=function(A,P){var C=this.j();try{A.call(P,C.resolve,C.reject)}catch(H){C.reject(H)}},l.prototype.then=function(A,P){function C(G,z){return typeof G=="function"?function(ne){try{H(G(ne))}catch(ue){K(ue)}}:z}var H,K,re=new l(function(G,z){H=G,K=z});return this.J(C(A,H),C(P,K)),re},l.prototype.catch=function(A){return this.then(void 0,A)},l.prototype.J=function(A,P){function C(){switch(H.h){case 1:A(H.i);break;case 2:P(H.i);break;default:throw Error("Unexpected state: "+H.h)}}var H=this;this.g==null?M.h(C):this.g.push(C),this.o=!0},l.resolve=m,l.reject=function(A){return new l(function(P,C){C(A)})},l.race=function(A){return new l(function(P,C){for(var H=c(A),K=H.next();!K.done;K=H.next())m(K.value).J(P,C)})},l.all=function(A){var P=c(A),C=P.next();return C.done?m([]):new l(function(H,K){function re(ne){return function(ue){G[ne]=ue,z--,z==0&&H(G)}}var G=[],z=0;do G.push(void 0),z++,m(C.value).J(re(G.length-1),K),C=P.next();while(!C.done)})},l});function V(n,l){n instanceof String&&(n+="");var d=0,m=!1,y={next:function(){if(!m&&d<n.length){var M=d++;return{value:l(M,n[M]),done:!1}}return m=!0,{done:!0,value:void 0}}};return y[Symbol.iterator]=function(){return y},y}var q=typeof Object.assign=="function"?Object.assign:function(n,l){for(var d=1;d<arguments.length;d++){var m=arguments[d];if(m)for(var y in m)Object.prototype.hasOwnProperty.call(m,y)&&(n[y]=m[y])}return n};t("Object.assign",function(n){return n||q}),t("Object.is",function(n){return n||function(l,d){return l===d?l!==0||1/l===1/d:l!==l&&d!==d}}),t("Array.prototype.includes",function(n){return n||function(l,d){var m=this;m instanceof String&&(m=String(m));var y=m.length;for(d=d||0,0>d&&(d=Math.max(d+y,0));d<y;d++){var M=m[d];if(M===l||Object.is(M,l))return!0}return!1}}),t("String.prototype.includes",function(n){return n||function(l,d){if(this==null)throw new TypeError("The 'this' value for String.prototype.includes must not be null or undefined");if(l instanceof RegExp)throw new TypeError("First argument to String.prototype.includes must not be a regular expression");return this.indexOf(l,d||0)!==-1}}),t("Array.prototype.keys",function(n){return n||function(){return V(this,function(l){return l})}});var $=this||self;function Y(n,l){n=n.split(".");var d=$;n[0]in d||typeof d.execScript>"u"||d.execScript("var "+n[0]);for(var m;n.length&&(m=n.shift());)n.length||l===void 0?d[m]&&d[m]!==Object.prototype[m]?d=d[m]:d=d[m]={}:d[m]=l}function Q(n){$.setTimeout(function(){throw n},0)}function ee(n){Q(n)}function ce(n,l){ee(Error("Invalid wire type: "+n+" (at position "+l+")"))}function ae(){ee(Error("Failed to read varint, encoding is invalid."))}function xe(n,l){return l=String.fromCharCode.apply(null,l),n==null?l:n+l}var Te,Ee=typeof TextDecoder<"u",qe,ke=typeof TextEncoder<"u";function yt(n){if(ke)n=(qe||(qe=new TextEncoder)).encode(n);else{var l=void 0;l=l===void 0?!1:l;for(var d=0,m=new Uint8Array(3*n.length),y=0;y<n.length;y++){var M=n.charCodeAt(y);if(128>M)m[d++]=M;else{if(2048>M)m[d++]=M>>6|192;else{if(55296<=M&&57343>=M){if(56319>=M&&y<n.length){var A=n.charCodeAt(++y);if(56320<=A&&57343>=A){M=1024*(M-55296)+A-56320+65536,m[d++]=M>>18|240,m[d++]=M>>12&63|128,m[d++]=M>>6&63|128,m[d++]=M&63|128;continue}else y--}if(l)throw Error("Found an unpaired surrogate");M=65533}m[d++]=M>>12|224,m[d++]=M>>6&63|128}m[d++]=M&63|128}}n=m.subarray(0,d)}return n}var bt={},nt=null;function It(n){var l;l===void 0&&(l=0),Nt(),l=bt[l];for(var d=Array(Math.floor(n.length/3)),m=l[64]||"",y=0,M=0;y<n.length-2;y+=3){var A=n[y],P=n[y+1],C=n[y+2],H=l[A>>2];A=l[(A&3)<<4|P>>4],P=l[(P&15)<<2|C>>6],C=l[C&63],d[M++]=H+A+P+C}switch(H=0,C=m,n.length-y){case 2:H=n[y+1],C=l[(H&15)<<2]||m;case 1:n=n[y],d[M]=l[n>>2]+l[(n&3)<<4|H>>4]+C+m}return d.join("")}function ge(n){var l=n.length,d=3*l/4;d%3?d=Math.floor(d):"=.".indexOf(n[l-1])!=-1&&(d="=.".indexOf(n[l-2])!=-1?d-2:d-1);var m=new Uint8Array(d),y=0;return at(n,function(M){m[y++]=M}),m.subarray(0,y)}function at(n,l){function d(C){for(;m<n.length;){var H=n.charAt(m++),K=nt[H];if(K!=null)return K;if(!/^[\s\xa0]*$/.test(H))throw Error("Unknown base64 encoding at char: "+H)}return C}Nt();for(var m=0;;){var y=d(-1),M=d(0),A=d(64),P=d(64);if(P===64&&y===-1)break;l(y<<2|M>>4),A!=64&&(l(M<<4&240|A>>2),P!=64&&l(A<<6&192|P))}}function Nt(){if(!nt){nt={};for(var n="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split(""),l=["+/=","+/","-_=","-_.","-_"],d=0;5>d;d++){var m=n.concat(l[d].split(""));bt[d]=m;for(var y=0;y<m.length;y++){var M=m[y];nt[M]===void 0&&(nt[M]=y)}}}}var xt=typeof Uint8Array=="function",Ke;function Ot(n){if(this.g=n,n!==null&&n.length===0)throw Error("ByteString should be constructed with non-empty values")}Ot.prototype.toJSON=function(){if(this.g==null)var n="";else n=this.g,n=this.g=n==null||typeof n=="string"?n:xt&&n instanceof Uint8Array?It(n):null;return n};var Xe=typeof Uint8Array.prototype.slice=="function";function st(n,l,d){return l===d?Ke||(Ke=new Uint8Array(0)):Xe?n.slice(l,d):new Uint8Array(n.subarray(l,d))}var he=0,je=0;function Zt(n){if(n.constructor===Uint8Array)return n;if(n.constructor===ArrayBuffer)return new Uint8Array(n);if(n.constructor===Array)return new Uint8Array(n);if(n.constructor===String)return ge(n);if(n.constructor===Ot){if(n.g==null)var l=Ke||(Ke=new Uint8Array(0));else{l=Uint8Array;var d=n.g;d=d==null||xt&&d!=null&&d instanceof Uint8Array?d:typeof d=="string"?ge(d):null,n=n.g=d,l=new l(n)}return l}if(n instanceof Uint8Array)return new Uint8Array(n.buffer,n.byteOffset,n.byteLength);throw Error("Type not convertible to a Uint8Array, expected a Uint8Array, an ArrayBuffer, a base64 encoded string, or Array of numbers")}function Pt(n,l){l=l===void 0?{}:l,l=l.u===void 0?!1:l.u,this.h=null,this.g=this.i=this.l=0,this.j=!1,this.u=l,n&&it(this,n)}function it(n,l){n.h=Zt(l),n.l=0,n.i=n.h.length,n.g=n.l}Pt.prototype.reset=function(){this.g=this.l};function Se(n){n.g>n.i&&(n.j=!0,ee(Error("Tried to read past the end of the data "+n.g+" > "+n.i)))}function ot(n){var l=n.h,d=l[n.g],m=d&127;return 128>d?(n.g+=1,Se(n),m):(d=l[n.g+1],m|=(d&127)<<7,128>d?(n.g+=2,Se(n),m):(d=l[n.g+2],m|=(d&127)<<14,128>d?(n.g+=3,Se(n),m):(d=l[n.g+3],m|=(d&127)<<21,128>d?(n.g+=4,Se(n),m):(d=l[n.g+4],m|=(d&15)<<28,128>d?(n.g+=5,Se(n),m>>>0):(n.g+=5,128<=l[n.g++]&&128<=l[n.g++]&&128<=l[n.g++]&&128<=l[n.g++]&&128<=l[n.g++]?(n.j=!0,ae(),m):(Se(n),m))))))}var Dt=[];function Bt(){this.g=new Uint8Array(64),this.h=0}function We(n,l){if(!(n.h+1<n.g.length)){var d=n.g;n.g=new Uint8Array(Math.ceil(1+2*n.g.length)),n.g.set(d)}n.g[n.h++]=l}Bt.prototype.length=function(){return this.h},Bt.prototype.end=function(){var n=this.g,l=this.h;return this.h=0,st(n,0,l)};function ut(n,l){for(;127<l;)We(n,l&127|128),l>>>=7;We(n,l)}function Ye(n){var l={},d=l.N===void 0?!1:l.N;this.m={u:l.u===void 0?!1:l.u},this.N=d,l=this.m,Dt.length?(d=Dt.pop(),l&&(d.u=l.u),n&&it(d,n),n=d):n=new Pt(n,l),this.g=n,this.l=this.g.g,this.h=this.i=-1,this.j=!1}Ye.prototype.reset=function(){this.g.reset(),this.h=this.i=-1};function lt(n){var l=n.g;if((l=l.g==l.i)||(l=n.j)||(l=n.g,l=l.j||0>l.g||l.g>l.i),l)return!1;n.l=n.g.g;var d=ot(n.g);return l=d>>>3,d&=7,0<=d&&5>=d?(n.i=l,n.h=d,!0):(n.j=!0,ce(d,n.l),!1)}function _t(n){switch(n.h){case 0:if(n.h!=0)_t(n);else e:{n=n.g;for(var l=n.g,d=0;10>d;d++){if(!(n.h[l]&128)){n.g=l+1,Se(n);break e}l++}n.j=!0,ae()}break;case 1:n=n.g,n.g+=8,Se(n);break;case 2:n.h!=2?_t(n):(l=ot(n.g),n=n.g,n.g+=l,Se(n));break;case 5:n=n.g,n.g+=4,Se(n);break;case 3:l=n.i;do{if(!lt(n)){n.j=!0,ee(Error("Unmatched start-group tag: stream EOF"));break}if(n.h==4){n.i!=l&&(n.j=!0,ee(Error("Unmatched end-group tag")));break}_t(n)}while(1);break;default:n.j=!0,ce(n.h,n.l)}}function en(n,l,d){n.N||(n=st(n.g.h,d,n.g.g),(d=l.m)?d.push(n):l.m=[n])}var Qe=[];function bn(){this.h=[],this.i=0,this.g=new Bt}function ct(n,l){l.length!==0&&(n.h.push(l),n.i+=l.length)}function vt(n,l,d){ut(n.g,8*l+2),ut(n.g,d.length),ct(n,n.g.end()),ct(n,d)}var _e=typeof Symbol=="function"&&typeof Symbol()=="symbol"?Symbol(void 0):void 0;function Lt(n,l){Object.isFrozen(n)||(_e?n[_e]|=l:n.g!==void 0?n.g|=l:Object.defineProperties(n,{g:{value:l,configurable:!0,writable:!0,enumerable:!1}}))}function St(n){if(!n)return 0;var l;return _e?l=n[_e]:l=n.g,l??0}function pt(n){return Array.isArray(n)&&Lt(n,1),n}function jt(n){if(!Array.isArray(n))throw Error("cannot mark non-array as immutable");Lt(n,2)}function Je(n){return n!==null&&typeof n=="object"&&n.constructor===Object}function Mt(n){switch(typeof n){case"number":return isFinite(n)?n:String(n);case"object":return xt&&n!=null&&n instanceof Uint8Array?It(n):n;default:return n}}function Wt(n,l){if(n!=null)return Array.isArray(n)||Je(n)?dt(n,l):l(n)}function dt(n,l){if(Array.isArray(n)){for(var d=Array(n.length),m=0;m<n.length;m++)d[m]=Wt(n[m],l);return St(n)&1&&pt(d),d}d={};for(m in n)d[m]=Wt(n[m],l);return d}var ht;function Re(n,l,d){var m=ht;ht=null,n||(n=m),m=this.constructor.ca,n||(n=m?[m]:[]),this.j=(m?0:-1)-(this.constructor.aa||0),this.i=null,this.g=n;e:{if(m=this.g.length,n=m-1,m&&(m=this.g[n],Je(m))){this.l=n-this.j,this.h=m;break e}l!==void 0&&-1<l?(this.l=Math.max(l,n+1-this.j),this.h=null):this.l=Number.MAX_VALUE}if(d)for(l=0;l<d.length;l++)n=d[l],n<this.l?(n+=this.j,(m=this.g[n])?pt(m):this.g[n]=F):(L(this),(m=this.h[n])?pt(m):this.h[n]=F)}var F=Object.freeze(pt([]));function L(n){var l=n.l+n.j;n.g[l]||(n.h=n.g[l]={})}function U(n,l,d){return l===-1?null:d!==void 0&&d||l>=n.l?n.h?n.h[l]:void 0:n.g[l+n.j]}function W(n,l,d){d=d===void 0?!0:d;var m=m===void 0?!1:m,y=U(n,l,m);return y==null&&(y=F),y===F?(y=pt([]),J(n,l,y,m)):d&&Array.isArray(y)&&St(y)&2&&(y=y.slice(),J(n,l,y,m)),y}function se(n,l,d){return n=U(n,l),n=n==null?n:+n,n??(d===void 0?0:d)}function J(n,l,d,m){m!==void 0&&m||l>=n.l?(L(n),n.h[l]=d):n.g[l+n.j]=d}function te(n,l,d){n.i||(n.i={});var m=n.i[d];if(!m){var y=W(n,d,!1);m=[];for(var M=Array.isArray(y)?!!(St(y)&2):!1,A=0;A<y.length;A++)m[A]=new l(y[A]),M&&jt(m[A].g);M&&(jt(m),Object.freeze(m)),n.i[d]=m}return m}function Z(n,l,d,m,y){var M=te(n,m,l);d=d||new m,n=W(n,l),y!=null?(M.splice(y,0,d),n.splice(y,0,pe(d))):(M.push(d),n.push(pe(d)))}Re.prototype.toJSON=function(){var n=pe(this);return dt(n,Mt)};function pe(n){if(n.i)for(var l in n.i){var d=n.i[l];if(Array.isArray(d))for(var m=0;m<d.length;m++)d[m]&&pe(d[m]);else d&&pe(d)}return n.g}Re.prototype.toString=function(){return pe(this).toString()};function Ce(n,l){return n=U(n,l),n??0}function Ze(n,l){return n=U(n,l),n??""}function At(n,l){if(n=n.m){ct(l,l.g.end());for(var d=0;d<n.length;d++)ct(l,n[d])}}function Ht(n){var l=n[0];switch(n.length){case 2:var d=n[1];return function(C,H,K){return l(C,H,K,d)};case 3:var m=n[1],y=n[2];return function(C,H,K){return l(C,H,K,m,y)};case 4:var M=n[1],A=n[2],P=n[3];return function(C,H,K){return l(C,H,K,M,A,P)};default:throw Error("unsupported number of parameters, expected [2-4], got "+n.length)}}function ft(n,l,d){for(;lt(l)&&l.h!=4;){var m=l.i,y=d[m];if(y){if(Array.isArray(y)&&(y=d[m]=Ht(y)),!y(l,n,m)){m=l,y=n;var M=m.l;_t(m),en(m,y,M)}}else m=l,y=n,M=m.l,_t(m),en(m,y,M)}return n}function ve(n,l){var d=new bn;if(l(n,d),n=d.i+d.g.length(),n===0)d=new Uint8Array(0);else{n=new Uint8Array(n);for(var m=d.h,y=m.length,M=l=0;M<y;M++){var A=m[M];A.length!==0&&(n.set(A,l),l+=A.length)}m=d.g,y=m.h,y!==0&&(n.set(m.g.subarray(0,y),l),m.h=0),d.h=[n],d=n}return d}function et(n,l,d){if(Qe.length){var m=Qe.pop();n&&(it(m.g,n),m.i=-1,m.h=-1),n=m}else n=new Ye(n);try{return d(new l,n)}finally{l=n.g,l.h=null,l.l=0,l.i=0,l.g=0,l.j=!1,l.u=!1,n.i=-1,n.h=-1,n.j=!1,100>Qe.length&&Qe.push(n)}}function Ne(n,l,d){if(l=U(l,d),l!=null){ut(n.g,8*d+5),n=n.g;var m=l;m=(d=0>m?1:0)?-m:m,m===0?0<1/m?he=je=0:(je=0,he=2147483648):isNaN(m)?(je=0,he=2147483647):34028234663852886e22<m?(je=0,he=(d<<31|2139095040)>>>0):11754943508222875e-54>m?(m=Math.round(m/Math.pow(2,-149)),je=0,he=(d<<31|m)>>>0):(l=Math.floor(Math.log(m)/Math.LN2),m*=Math.pow(2,-l),m=Math.round(8388608*m),16777216<=m&&++l,je=0,he=(d<<31|l+127<<23|m&8388607)>>>0),d=he,We(n,d>>>0&255),We(n,d>>>8&255),We(n,d>>>16&255),We(n,d>>>24&255)}}function $e(n,l,d){if(n.h!==5)return!1;n=n.g;var m=n.h[n.g],y=n.h[n.g+1],M=n.h[n.g+2],A=n.h[n.g+3];return n.g+=4,Se(n),y=(m<<0|y<<8|M<<16|A<<24)>>>0,n=2*(y>>31)+1,m=y>>>23&255,y&=8388607,J(l,d,m==255?y?NaN:1/0*n:m==0?n*Math.pow(2,-149)*y:n*Math.pow(2,m-150)*(y+Math.pow(2,23))),!0}function _n(n,l,d){if(n.h!==0)return!1;for(var m=n.g,y=128,M=0,A=n=0;4>A&&128<=y;A++)y=m.h[m.g++],M|=(y&127)<<7*A;if(128<=y&&(y=m.h[m.g++],M|=(y&127)<<28,n|=(y&127)>>4),128<=y)for(A=0;5>A&&128<=y;A++)y=m.h[m.g++],n|=(y&127)<<7*A+3;return 128>y?(m=M>>>0,y=n>>>0,(n=y&2147483648)&&(m=~m+1>>>0,y=~y>>>0,m==0&&(y=y+1>>>0)),m=4294967296*y+(m>>>0),n=n?-m:m):(m.j=!0,ae(),n=void 0),J(l,d,n),!0}function sr(n,l,d){return n.h!==0?!1:(J(l,d,ot(n.g)),!0)}function Dn(n,l,d){if(n.h!==2)return!1;var m=ot(n.g);n=n.g;var y=n.g;n.g+=m,Se(n),n=n.h;var M;if(Ee)(M=Te)||(M=Te=new TextDecoder("utf-8",{fatal:!1})),M=M.decode(n.subarray(y,y+m));else{m=y+m;for(var A=[],P=null,C,H,K;y<m;)C=n[y++],128>C?A.push(C):224>C?y>=m?A.push(65533):(H=n[y++],194>C||(H&192)!==128?(y--,A.push(65533)):A.push((C&31)<<6|H&63)):240>C?y>=m-1?A.push(65533):(H=n[y++],(H&192)!==128||C===224&&160>H||C===237&&160<=H||((M=n[y++])&192)!==128?(y--,A.push(65533)):A.push((C&15)<<12|(H&63)<<6|M&63)):244>=C?y>=m-2?A.push(65533):(H=n[y++],(H&192)!==128||(C<<28)+(H-144)>>30||((M=n[y++])&192)!==128||((K=n[y++])&192)!==128?(y--,A.push(65533)):(C=(C&7)<<18|(H&63)<<12|(M&63)<<6|K&63,C-=65536,A.push((C>>10&1023)+55296,(C&1023)+56320))):A.push(65533),8192<=A.length&&(P=xe(P,A),A.length=0);M=xe(P,A)}return J(l,d,M),!0}function Bn(n,l,d,m,y){if(n.h!==2)return!1;var M=new m,A=n.g.i,P=ot(n.g),C=n.g.g+P;if(n.g.i=C,y(M,n),y=C-n.g.g,y!==0)throw Error("Message parsing ended unexpectedly. Expected to read "+(P+" bytes, instead read "+(P-y)+" bytes, either the data ended unexpectedly or the message misreported its own length"));return n.g.g=C,n.g.i=A,Z(l,d,M,m,void 0),!0}function rt(n){Re.call(this,n)}var Ln;v(rt,Re);function ir(n,l){var d=U(n,1);if(d!=null&&d!=null){ut(l.g,8);var m=l.g;if(0<=d)ut(m,d);else{for(var y=0;9>y;y++)We(m,d&127|128),d>>=7;We(m,1)}}Ne(l,n,2),m=U(n,3),m!=null&&vt(l,3,yt(m)),m=U(n,4),m!=null&&vt(l,4,yt(m)),At(n,l)}function jn(n,l){return ft(n,l,Ln||(Ln={1:sr,2:$e,3:Dn,4:Dn}))}function vn(n){Re.call(this,n,-1,ur)}var tn;v(vn,Re),vn.prototype.addClassification=function(n,l){return Z(this,1,n,rt,l),this};function or(n,l){return ft(n,l,tn||(tn={1:[Bn,rt,jn]}))}var ur=[1];function nn(n){Re.call(this,n)}var wt;v(nn,Re);function wn(n,l){Ne(l,n,1),Ne(l,n,2),Ne(l,n,3),Ne(l,n,4),Ne(l,n,5),At(n,l)}function lr(n,l){return ft(n,l,wt||(wt={1:$e,2:$e,3:$e,4:$e,5:$e}))}function Wn(n){Re.call(this,n,-1,Vn)}var Hn;v(Wn,Re);function cr(n,l){return ft(n,l,Hn||(Hn={1:[Bn,nn,lr]}))}var Vn=[1];function rn(n){Re.call(this,n)}var Un;v(rn,Re);function pr(n,l){Ne(l,n,1),Ne(l,n,2),Ne(l,n,3),Ne(l,n,4),Ne(l,n,5);var d=U(n,6);if(d!=null&&d!=null){ut(l.g,48);var m=l.g,y=d;d=0>y,y=Math.abs(y);var M=y>>>0;for(y=Math.floor((y-M)/4294967296),y>>>=0,d&&(y=~y>>>0,M=(~M>>>0)+1,4294967295<M&&(M=0,y++,4294967295<y&&(y=0))),he=M,je=y,d=he,M=je;0<M||127<d;)We(m,d&127|128),d=(d>>>7|M<<25)>>>0,M>>>=7;We(m,d)}At(n,l)}function zn(n,l){return ft(n,l,Un||(Un={1:$e,2:$e,3:$e,4:$e,5:$e,6:_n}))}function kn(n,l,d){if(d=n.createShader(d===0?n.VERTEX_SHADER:n.FRAGMENT_SHADER),n.shaderSource(d,l),n.compileShader(d),!n.getShaderParameter(d,n.COMPILE_STATUS))throw Error(`Could not compile WebGL shader.

`+n.getShaderInfoLog(d));return d}function dr(n){return te(n,rt,1).map(function(l){return{index:Ce(l,1),X:se(l,2),label:U(l,3)!=null?Ze(l,3):void 0,displayName:U(l,4)!=null?Ze(l,4):void 0}})}function hr(n){return{x:se(n,1),y:se(n,2),z:se(n,3),visibility:U(n,4)!=null?se(n,4):void 0}}function In(n){return te(et(n,Wn,cr),nn,1).map(hr)}function xn(n,l){this.h=n,this.g=l,this.l=0}function Sn(n,l,d){return an(n,l),typeof n.g.canvas.transferToImageBitmap=="function"?Promise.resolve(n.g.canvas.transferToImageBitmap()):d?Promise.resolve(n.g.canvas):typeof createImageBitmap=="function"?createImageBitmap(n.g.canvas):(n.i===void 0&&(n.i=document.createElement("canvas")),new Promise(function(m){n.i.height=n.g.canvas.height,n.i.width=n.g.canvas.width,n.i.getContext("2d",{}).drawImage(n.g.canvas,0,0,n.g.canvas.width,n.g.canvas.height),m(n.i)}))}function an(n,l){var d=n.g;if(n.m===void 0){var m=kn(d,`
  attribute vec2 aVertex;
  attribute vec2 aTex;
  varying vec2 vTex;
  void main(void) {
    gl_Position = vec4(aVertex, 0.0, 1.0);
    vTex = aTex;
  }`,0),y=kn(d,`
  precision mediump float;
  varying vec2 vTex;
  uniform sampler2D sampler0;
  void main(){
    gl_FragColor = texture2D(sampler0, vTex);
  }`,1),M=d.createProgram();if(d.attachShader(M,m),d.attachShader(M,y),d.linkProgram(M),!d.getProgramParameter(M,d.LINK_STATUS))throw Error(`Could not compile WebGL program.

`+d.getProgramInfoLog(M));m=n.m=M,d.useProgram(m),y=d.getUniformLocation(m,"sampler0"),n.j={I:d.getAttribLocation(m,"aVertex"),H:d.getAttribLocation(m,"aTex"),da:y},n.s=d.createBuffer(),d.bindBuffer(d.ARRAY_BUFFER,n.s),d.enableVertexAttribArray(n.j.I),d.vertexAttribPointer(n.j.I,2,d.FLOAT,!1,0,0),d.bufferData(d.ARRAY_BUFFER,new Float32Array([-1,-1,-1,1,1,1,1,-1]),d.STATIC_DRAW),d.bindBuffer(d.ARRAY_BUFFER,null),n.o=d.createBuffer(),d.bindBuffer(d.ARRAY_BUFFER,n.o),d.enableVertexAttribArray(n.j.H),d.vertexAttribPointer(n.j.H,2,d.FLOAT,!1,0,0),d.bufferData(d.ARRAY_BUFFER,new Float32Array([0,1,0,0,1,0,1,1]),d.STATIC_DRAW),d.bindBuffer(d.ARRAY_BUFFER,null),d.uniform1i(y,0)}m=n.j,d.useProgram(n.m),d.canvas.width=l.width,d.canvas.height=l.height,d.viewport(0,0,l.width,l.height),d.activeTexture(d.TEXTURE0),n.h.bindTexture2d(l.glName),d.enableVertexAttribArray(m.I),d.bindBuffer(d.ARRAY_BUFFER,n.s),d.vertexAttribPointer(m.I,2,d.FLOAT,!1,0,0),d.enableVertexAttribArray(m.H),d.bindBuffer(d.ARRAY_BUFFER,n.o),d.vertexAttribPointer(m.H,2,d.FLOAT,!1,0,0),d.bindFramebuffer(d.DRAW_FRAMEBUFFER?d.DRAW_FRAMEBUFFER:d.FRAMEBUFFER,null),d.clearColor(0,0,0,0),d.clear(d.COLOR_BUFFER_BIT),d.colorMask(!0,!0,!0,!0),d.drawArrays(d.TRIANGLE_FAN,0,4),d.disableVertexAttribArray(m.I),d.disableVertexAttribArray(m.H),d.bindBuffer(d.ARRAY_BUFFER,null),n.h.bindTexture2d(0)}function $n(n){this.g=n}var Gn=new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,10,9,1,7,0,65,0,253,15,26,11]);function qn(n,l){return l+n}function sn(n,l){window[n]=l}function Kn(n){var l=document.createElement("script");return l.setAttribute("src",n),l.setAttribute("crossorigin","anonymous"),new Promise(function(d){l.addEventListener("load",function(){d()},!1),l.addEventListener("error",function(){d()},!1),document.body.appendChild(l)})}function Xn(){return D(function(n){switch(n.g){case 1:return n.m=2,I(n,WebAssembly.instantiate(Gn),4);case 4:n.g=3,n.m=0;break;case 2:return n.m=0,n.j=null,n.return(!1);case 3:return n.return(!0)}})}function Vt(n){if(this.g=n,this.listeners={},this.j={},this.F={},this.m={},this.s={},this.G=this.o=this.R=!0,this.C=Promise.resolve(),this.P="",this.B={},this.locateFile=n&&n.locateFile||qn,typeof window=="object")var l=window.location.pathname.toString().substring(0,window.location.pathname.toString().lastIndexOf("/"))+"/";else if(typeof location<"u")l=location.pathname.toString().substring(0,location.pathname.toString().lastIndexOf("/"))+"/";else throw Error("solutions can only be loaded on a web page or in a web worker");if(this.S=l,n.options){l=c(Object.keys(n.options));for(var d=l.next();!d.done;d=l.next()){d=d.value;var m=n.options[d].default;m!==void 0&&(this.j[d]=typeof m=="function"?m():m)}}}s=Vt.prototype,s.close=function(){return this.i&&this.i.delete(),Promise.resolve()};function on(n){var l,d,m,y,M,A,P,C,H,K,re;return D(function(G){switch(G.g){case 1:return n.R?(l=n.g.files===void 0?[]:typeof n.g.files=="function"?n.g.files(n.j):n.g.files,I(G,Xn(),2)):G.return();case 2:if(d=G.h,typeof window=="object")return sn("createMediapipeSolutionsWasm",{locateFile:n.locateFile}),sn("createMediapipeSolutionsPackedAssets",{locateFile:n.locateFile}),A=l.filter(function(z){return z.data!==void 0}),P=l.filter(function(z){return z.data===void 0}),C=Promise.all(A.map(function(z){var ne=Ut(n,z.url);if(z.path!==void 0){var ue=z.path;ne=ne.then(function(ye){return n.overrideFile(ue,ye),Promise.resolve(ye)})}return ne})),H=Promise.all(P.map(function(z){return z.simd===void 0||z.simd&&d||!z.simd&&!d?Kn(n.locateFile(z.url,n.S)):Promise.resolve()})).then(function(){var z,ne,ue;return D(function(ye){if(ye.g==1)return z=window.createMediapipeSolutionsWasm,ne=window.createMediapipeSolutionsPackedAssets,ue=n,I(ye,z(ne),2);ue.h=ye.h,ye.g=0})}),K=function(){return D(function(z){return n.g.graph&&n.g.graph.url?z=I(z,Ut(n,n.g.graph.url),0):(z.g=0,z=void 0),z})}(),I(G,Promise.all([H,C,K]),7);if(typeof importScripts!="function")throw Error("solutions can only be loaded on a web page or in a web worker");return m=l.filter(function(z){return z.simd===void 0||z.simd&&d||!z.simd&&!d}).map(function(z){return n.locateFile(z.url,n.S)}),importScripts.apply(null,p(m)),y=n,I(G,createMediapipeSolutionsWasm(Module),6);case 6:y.h=G.h,n.l=new OffscreenCanvas(1,1),n.h.canvas=n.l,M=n.h.GL.createContext(n.l,{antialias:!1,alpha:!1,ba:typeof WebGL2RenderingContext<"u"?2:1}),n.h.GL.makeContextCurrent(M),G.g=4;break;case 7:if(n.l=document.createElement("canvas"),re=n.l.getContext("webgl2",{}),!re&&(re=n.l.getContext("webgl",{}),!re))return alert("Failed to create WebGL canvas context when passing video frame."),G.return();n.D=re,n.h.canvas=n.l,n.h.createContext(n.l,!0,!0,{});case 4:n.i=new n.h.SolutionWasm,n.R=!1,G.g=0}})}function un(n){var l,d,m,y,M,A,P,C;return D(function(H){if(H.g==1){if(n.g.graph&&n.g.graph.url&&n.P===n.g.graph.url)return H.return();if(n.o=!0,!n.g.graph||!n.g.graph.url){H.g=2;return}return n.P=n.g.graph.url,I(H,Ut(n,n.g.graph.url),3)}for(H.g!=2&&(l=H.h,n.i.loadGraph(l)),d=c(Object.keys(n.B)),m=d.next();!m.done;m=d.next())y=m.value,n.i.overrideFile(y,n.B[y]);if(n.B={},n.g.listeners)for(M=c(n.g.listeners),A=M.next();!A.done;A=M.next())P=A.value,fr(n,P);C=n.j,n.j={},n.setOptions(C),H.g=0})}s.reset=function(){var n=this;return D(function(l){n.i&&(n.i.reset(),n.m={},n.s={}),l.g=0})},s.setOptions=function(n,l){var d=this;if(l=l||this.g.options){for(var m=[],y=[],M={},A=c(Object.keys(n)),P=A.next();!P.done;M={K:M.K,L:M.L},P=A.next()){var C=P.value;C in this.j&&this.j[C]===n[C]||(this.j[C]=n[C],P=l[C],P!==void 0&&(P.onChange&&(M.K=P.onChange,M.L=n[C],m.push(function(H){return function(){var K;return D(function(re){if(re.g==1)return I(re,H.K(H.L),2);K=re.h,K===!0&&(d.o=!0),re.g=0})}}(M))),P.graphOptionXref&&(C={valueNumber:P.type===1?n[C]:0,valueBoolean:P.type===0?n[C]:!1,valueString:P.type===2?n[C]:""},P=Object.assign(Object.assign(Object.assign({},{calculatorName:"",calculatorIndex:0}),P.graphOptionXref),C),y.push(P))))}(m.length!==0||y.length!==0)&&(this.o=!0,this.A=(this.A===void 0?[]:this.A).concat(y),this.v=(this.v===void 0?[]:this.v).concat(m))}};function ln(n){var l,d,m,y,M,A,P;return D(function(C){switch(C.g){case 1:if(!n.o)return C.return();if(!n.v){C.g=2;break}l=c(n.v),d=l.next();case 3:if(d.done){C.g=5;break}return m=d.value,I(C,m(),4);case 4:d=l.next(),C.g=3;break;case 5:n.v=void 0;case 2:if(n.A){for(y=new n.h.GraphOptionChangeRequestList,M=c(n.A),A=M.next();!A.done;A=M.next())P=A.value,y.push_back(P);n.i.changeOptions(y),y.delete(),n.A=void 0}n.o=!1,C.g=0}})}s.initialize=function(){var n=this;return D(function(l){return l.g==1?I(l,on(n),2):l.g!=3?I(l,un(n),3):I(l,ln(n),0)})};function Ut(n,l){var d,m;return D(function(y){return l in n.F?y.return(n.F[l]):(d=n.locateFile(l,""),m=fetch(d).then(function(M){return M.arrayBuffer()}),n.F[l]=m,y.return(m))})}s.overrideFile=function(n,l){this.i?this.i.overrideFile(n,l):this.B[n]=l},s.clearOverriddenFiles=function(){this.B={},this.i&&this.i.clearOverriddenFiles()},s.send=function(n,l){var d=this,m,y,M,A,P,C,H,K,re;return D(function(G){switch(G.g){case 1:return d.g.inputs?(m=1e3*(l??performance.now()),I(G,d.C,2)):G.return();case 2:return I(G,d.initialize(),3);case 3:for(y=new d.h.PacketDataList,M=c(Object.keys(n)),A=M.next();!A.done;A=M.next())if(P=A.value,C=d.g.inputs[P]){e:{var z=n[P];switch(C.type){case"video":var ne=d.m[C.stream];if(ne||(ne=new xn(d.h,d.D),d.m[C.stream]=ne),ne.l===0&&(ne.l=ne.h.createTexture()),typeof HTMLVideoElement<"u"&&z instanceof HTMLVideoElement)var ue=z.videoWidth,ye=z.videoHeight;else typeof HTMLImageElement<"u"&&z instanceof HTMLImageElement?(ue=z.naturalWidth,ye=z.naturalHeight):(ue=z.width,ye=z.height);ye={glName:ne.l,width:ue,height:ye},ue=ne.g,ue.canvas.width=ye.width,ue.canvas.height=ye.height,ue.activeTexture(ue.TEXTURE0),ne.h.bindTexture2d(ne.l),ue.texImage2D(ue.TEXTURE_2D,0,ue.RGBA,ue.RGBA,ue.UNSIGNED_BYTE,z),ne.h.bindTexture2d(0),ne=ye;break e;case"detections":for(ne=d.m[C.stream],ne||(ne=new $n(d.h),d.m[C.stream]=ne),ne.data||(ne.data=new ne.g.DetectionListData),ne.data.reset(z.length),ye=0;ye<z.length;++ye){ue=z[ye];var fe=ne.data,Fe=fe.setBoundingBox,tt=ye,He=ue.T,le=new rn;if(J(le,1,He.Y),J(le,2,He.Z),J(le,3,He.height),J(le,4,He.width),J(le,5,He.rotation),J(le,6,He.W),He=ve(le,pr),Fe.call(fe,tt,He),ue.O)for(fe=0;fe<ue.O.length;++fe){le=ue.O[fe];var Ie=!!le.visibility;Fe=ne.data,tt=Fe.addNormalizedLandmark,He=ye,le=Object.assign(Object.assign({},le),{visibility:Ie?le.visibility:0}),Ie=new nn,J(Ie,1,le.x),J(Ie,2,le.y),J(Ie,3,le.z),le.visibility&&J(Ie,4,le.visibility),le=ve(Ie,wn),tt.call(Fe,He,le)}if(ue.M)for(fe=0;fe<ue.M.length;++fe)Fe=ne.data,tt=Fe.addClassification,He=ye,le=ue.M[fe],Ie=new rt,J(Ie,2,le.X),le.index&&J(Ie,1,le.index),le.label&&J(Ie,3,le.label),le.displayName&&J(Ie,4,le.displayName),le=ve(Ie,ir),tt.call(Fe,He,le)}ne=ne.data;break e;default:ne={}}}switch(H=ne,K=C.stream,C.type){case"video":y.pushTexture2d(Object.assign(Object.assign({},H),{stream:K,timestamp:m}));break;case"detections":re=H,re.stream=K,re.timestamp=m,y.pushDetectionList(re);break;default:throw Error("Unknown input config type: '"+C.type+"'")}}return d.i.send(y),I(G,d.C,4);case 4:y.delete(),G.g=0}})};function cn(n,l,d){var m,y,M,A,P,C,H,K,re,G,z,ne,ue,ye;return D(function(fe){switch(fe.g){case 1:if(!d)return fe.return(l);for(m={},y=0,M=c(Object.keys(d)),A=M.next();!A.done;A=M.next())P=A.value,C=d[P],typeof C!="string"&&C.type==="texture"&&l[C.stream]!==void 0&&++y;1<y&&(n.G=!1),H=c(Object.keys(d)),A=H.next();case 2:if(A.done){fe.g=4;break}if(K=A.value,re=d[K],typeof re=="string")return ue=m,ye=K,I(fe,pn(n,K,l[re]),14);if(G=l[re.stream],re.type==="detection_list"){if(G){for(var Fe=G.getRectList(),tt=G.getLandmarksList(),He=G.getClassificationsList(),le=[],Ie=0;Ie<Fe.size();++Ie){var Tt=et(Fe.get(Ie),rn,zn);Tt={T:{Y:se(Tt,1),Z:se(Tt,2),height:se(Tt,3),width:se(Tt,4),rotation:se(Tt,5,0),W:Ce(Tt,6)},O:In(tt.get(Ie)),M:dr(et(He.get(Ie),vn,or))},le.push(Tt)}Fe=le}else Fe=[];m[K]=Fe,fe.g=7;break}if(re.type==="proto_list"){if(G){for(Fe=Array(G.size()),tt=0;tt<G.size();tt++)Fe[tt]=G.get(tt);G.delete()}else Fe=[];m[K]=Fe,fe.g=7;break}if(G===void 0){fe.g=3;break}if(re.type==="float_list"){m[K]=G,fe.g=7;break}if(re.type==="proto"){m[K]=G,fe.g=7;break}if(re.type!=="texture")throw Error("Unknown output config type: '"+re.type+"'");return z=n.s[K],z||(z=new xn(n.h,n.D),n.s[K]=z),I(fe,Sn(z,G,n.G),13);case 13:ne=fe.h,m[K]=ne;case 7:re.transform&&m[K]&&(m[K]=re.transform(m[K])),fe.g=3;break;case 14:ue[ye]=fe.h;case 3:A=H.next(),fe.g=2;break;case 4:return fe.return(m)}})}function pn(n,l,d){var m;return D(function(y){return typeof d=="number"||d instanceof Uint8Array||d instanceof n.h.Uint8BlobList?y.return(d):d instanceof n.h.Texture2dDataOut?(m=n.s[l],m||(m=new xn(n.h,n.D),n.s[l]=m),y.return(Sn(m,d,n.G))):y.return(void 0)})}function fr(n,l){for(var d=l.name||"$",m=[].concat(p(l.wants)),y=new n.h.StringList,M=c(l.wants),A=M.next();!A.done;A=M.next())y.push_back(A.value);M=n.h.PacketListener.implement({onResults:function(P){for(var C={},H=0;H<l.wants.length;++H)C[m[H]]=P.get(H);var K=n.listeners[d];K&&(n.C=cn(n,C,l.outs).then(function(re){re=K(re);for(var G=0;G<l.wants.length;++G){var z=C[m[G]];typeof z=="object"&&z.hasOwnProperty&&z.hasOwnProperty("delete")&&z.delete()}re&&(n.C=re)}))}}),n.i.attachMultiListener(y,M),y.delete()}s.onResults=function(n,l){this.listeners[l||"$"]=n},Y("Solution",Vt),Y("OptionType",{BOOL:0,NUMBER:1,$:2,0:"BOOL",1:"NUMBER",2:"STRING"});function w(n){switch(n===void 0&&(n=0),n){case 1:return"pose_landmark_full.tflite";case 2:return"pose_landmark_heavy.tflite";default:return"pose_landmark_lite.tflite"}}function T(n){var l=this;n=n||{},this.g=new Vt({locateFile:n.locateFile,files:function(d){return[{url:"pose_solution_packed_assets_loader.js"},{simd:!1,url:"pose_solution_wasm_bin.js"},{simd:!0,url:"pose_solution_simd_wasm_bin.js"},{data:!0,url:w(d.modelComplexity)}]},graph:{url:"pose_web.binarypb"},listeners:[{wants:["pose_landmarks","world_landmarks","segmentation_mask","image_transformed"],outs:{image:{type:"texture",stream:"image_transformed"},poseLandmarks:{type:"proto",stream:"pose_landmarks",transform:In},poseWorldLandmarks:{type:"proto",stream:"world_landmarks",transform:In},segmentationMask:{type:"texture",stream:"segmentation_mask"}}}],inputs:{image:{type:"video",stream:"input_frames_gpu"}},options:{useCpuInference:{type:0,graphOptionXref:{calculatorType:"InferenceCalculator",fieldName:"use_cpu_inference"},default:"iPad Simulator;iPhone Simulator;iPod Simulator;iPad;iPhone;iPod".split(";").includes(navigator.platform)||navigator.userAgent.includes("Mac")&&"ontouchend"in document},selfieMode:{type:0,graphOptionXref:{calculatorType:"GlScalerCalculator",calculatorIndex:1,fieldName:"flip_horizontal"}},modelComplexity:{type:1,graphOptionXref:{calculatorType:"ConstantSidePacketCalculator",calculatorName:"ConstantSidePacketCalculatorModelComplexity",fieldName:"int_value"},onChange:function(d){var m,y,M;return D(function(A){return A.g==1?(m=w(d),y="third_party/mediapipe/modules/pose_landmark/"+m,I(A,Ut(l.g,m),2)):(M=A.h,l.g.overrideFile(y,M),A.return(!0))})}},smoothLandmarks:{type:0,graphOptionXref:{calculatorType:"ConstantSidePacketCalculator",calculatorName:"ConstantSidePacketCalculatorSmoothLandmarks",fieldName:"bool_value"}},enableSegmentation:{type:0,graphOptionXref:{calculatorType:"ConstantSidePacketCalculator",calculatorName:"ConstantSidePacketCalculatorEnableSegmentation",fieldName:"bool_value"}},smoothSegmentation:{type:0,graphOptionXref:{calculatorType:"ConstantSidePacketCalculator",calculatorName:"ConstantSidePacketCalculatorSmoothSegmentation",fieldName:"bool_value"}},minDetectionConfidence:{type:1,graphOptionXref:{calculatorType:"TensorsToDetectionsCalculator",calculatorName:"poselandmarkgpu__posedetectiongpu__TensorsToDetectionsCalculator",fieldName:"min_score_thresh"}},minTrackingConfidence:{type:1,graphOptionXref:{calculatorType:"ThresholdingCalculator",calculatorName:"poselandmarkgpu__poselandmarkbyroigpu__tensorstoposelandmarksandsegmentation__ThresholdingCalculator",fieldName:"threshold"}}}})}s=T.prototype,s.reset=function(){this.g.reset()},s.close=function(){return this.g.close(),Promise.resolve()},s.onResults=function(n){this.g.onResults(n)},s.initialize=function(){var n=this;return D(function(l){return I(l,n.g.initialize(),0)})},s.send=function(n,l){var d=this;return D(function(m){return I(m,d.g.send(n,l),0)})},s.setOptions=function(n){this.g.setOptions(n)},Y("Pose",T),Y("POSE_CONNECTIONS",[[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32]]),Y("POSE_LANDMARKS",{NOSE:0,LEFT_EYE_INNER:1,LEFT_EYE:2,LEFT_EYE_OUTER:3,RIGHT_EYE_INNER:4,RIGHT_EYE:5,RIGHT_EYE_OUTER:6,LEFT_EAR:7,RIGHT_EAR:8,LEFT_RIGHT:9,RIGHT_LEFT:10,LEFT_SHOULDER:11,RIGHT_SHOULDER:12,LEFT_ELBOW:13,RIGHT_ELBOW:14,LEFT_WRIST:15,RIGHT_WRIST:16,LEFT_PINKY:17,RIGHT_PINKY:18,LEFT_INDEX:19,RIGHT_INDEX:20,LEFT_THUMB:21,RIGHT_THUMB:22,LEFT_HIP:23,RIGHT_HIP:24,LEFT_KNEE:25,RIGHT_KNEE:26,LEFT_ANKLE:27,RIGHT_ANKLE:28,LEFT_HEEL:29,RIGHT_HEEL:30,LEFT_FOOT_INDEX:31,RIGHT_FOOT_INDEX:32}),Y("POSE_LANDMARKS_LEFT",{LEFT_EYE_INNER:1,LEFT_EYE:2,LEFT_EYE_OUTER:3,LEFT_EAR:7,LEFT_RIGHT:9,LEFT_SHOULDER:11,LEFT_ELBOW:13,LEFT_WRIST:15,LEFT_PINKY:17,LEFT_INDEX:19,LEFT_THUMB:21,LEFT_HIP:23,LEFT_KNEE:25,LEFT_ANKLE:27,LEFT_HEEL:29,LEFT_FOOT_INDEX:31}),Y("POSE_LANDMARKS_RIGHT",{RIGHT_EYE_INNER:4,RIGHT_EYE:5,RIGHT_EYE_OUTER:6,RIGHT_EAR:8,RIGHT_LEFT:10,RIGHT_SHOULDER:12,RIGHT_ELBOW:14,RIGHT_WRIST:16,RIGHT_PINKY:18,RIGHT_INDEX:20,RIGHT_THUMB:22,RIGHT_HIP:24,RIGHT_KNEE:26,RIGHT_ANKLE:28,RIGHT_HEEL:30,RIGHT_FOOT_INDEX:32}),Y("POSE_LANDMARKS_NEUTRAL",{NOSE:0}),Y("VERSION","0.5.1635988162")}).call(Tn);/**
    * @license
    * Copyright 2022 Google LLC. All Rights Reserved.
    * Licensed under the Apache License, Version 2.0 (the "License");
    * you may not use this file except in compliance with the License.
    * You may obtain a copy of the License at
    *
    * http://www.apache.org/licenses/LICENSE-2.0
    *
    * Unless required by applicable law or agreed to in writing, software
    * distributed under the License is distributed on an "AS IS" BASIS,
    * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    * See the License for the specific language governing permissions and
    * limitations under the License.
    * =============================================================================
    */var Hi=function(s,e){return(Hi=Object.setPrototypeOf||{__proto__:[]}instanceof Array&&function(a,o){a.__proto__=o}||function(a,o){for(var i in o)o.hasOwnProperty(i)&&(a[i]=o[i])})(s,e)};function Vi(s,e){function a(){this.constructor=s}Hi(s,e),s.prototype=e===null?Object.create(e):(a.prototype=e.prototype,new a)}var de=function(){return(de=Object.assign||function(s){for(var e,a=1,o=arguments.length;a<o;a++)for(var i in e=arguments[a])Object.prototype.hasOwnProperty.call(e,i)&&(s[i]=e[i]);return s}).apply(this,arguments)};function ie(s,e,a,o){return new(a||(a=Promise))(function(i,t){function u(h){try{p(o.next(h))}catch(r){t(r)}}function c(h){try{p(o.throw(h))}catch(r){t(r)}}function p(h){var r;h.done?i(h.value):(r=h.value,r instanceof a?r:new a(function(f){f(r)})).then(u,c)}p((o=o.apply(s,e||[])).next())})}function oe(s,e){var a,o,i,t,u={label:0,sent:function(){if(1&i[0])throw i[1];return i[1]},trys:[],ops:[]};return t={next:c(0),throw:c(1),return:c(2)},typeof Symbol=="function"&&(t[Symbol.iterator]=function(){return this}),t;function c(p){return function(h){return function(r){if(a)throw new TypeError("Generator is already executing.");for(;u;)try{if(a=1,o&&(i=2&r[0]?o.return:r[0]?o.throw||((i=o.return)&&i.call(o),0):o.next)&&!(i=i.call(o,r[1])).done)return i;switch(o=0,i&&(r=[2&r[0],i.value]),r[0]){case 0:case 1:i=r;break;case 4:return u.label++,{value:r[1],done:!1};case 5:u.label++,o=r[1],r=[0];continue;case 7:r=u.ops.pop(),u.trys.pop();continue;default:if(i=u.trys,!((i=i.length>0&&i[i.length-1])||r[0]!==6&&r[0]!==2)){u=0;continue}if(r[0]===3&&(!i||r[1]>i[0]&&r[1]<i[3])){u.label=r[1];break}if(r[0]===6&&u.label<i[1]){u.label=i[1],i=r;break}if(i&&u.label<i[2]){u.label=i[2],u.ops.push(r);break}i[2]&&u.ops.pop(),u.trys.pop();continue}r=e.call(s,u)}catch(f){r=[6,f],o=0}finally{a=i=0}if(5&r[0])throw r[1];return{value:r[0]?r[1]:void 0,done:!0}}([p,h])}}}function Gt(){for(var s=0,e=0,a=arguments.length;e<a;e++)s+=arguments[e].length;var o=Array(s),i=0;for(e=0;e<a;e++)for(var t=arguments[e],u=0,c=t.length;u<c;u++,i++)o[i]=t[u];return o}var gt=["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],Cn=["nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer","left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index"];function nr(s){return s instanceof SVGAnimatedLength?s.baseVal.value:s}function Ui(s){return ie(this,void 0,void 0,function(){var e,a;return oe(this,function(o){switch(o.label){case 0:return e=document.createElement("canvas"),s instanceof gn?[4,Hr(s,e)]:[3,2];case 1:return o.sent(),[3,3];case 2:e.width=nr(s.width),e.height=nr(s.height),a=e.getContext("2d"),s instanceof ImageData?a.putImageData(s,0,0):a.drawImage(s,0,0),o.label=3;case 3:return[2,e]}})})}function zi(s){return ie(this,void 0,void 0,function(){var e,a,o,i,t,u;return oe(this,function(c){switch(c.label){case 0:return s instanceof gn?(e=s.shape.slice(0,2),a=e[0],o=e[1],i=ImageData.bind,[4,Hr(s)]):[3,2];case 1:return[2,new(i.apply(ImageData,[void 0,c.sent(),o,a]))];case 2:return t=document.createElement("canvas"),u=t.getContext("2d"),t.width=nr(s.width),t.height=nr(s.height),u.drawImage(s,0,0),[2,u.getImageData(0,0,t.width,t.height)]}})})}function Qh(s){return ie(this,void 0,void 0,function(){var e,a;return oe(this,function(o){switch(o.label){case 0:return s instanceof SVGImageElement||s instanceof OffscreenCanvas?[4,Ui(s)]:[3,2];case 1:return a=o.sent(),[3,3];case 2:a=s,o.label=3;case 3:return e=a,[2,Os(e,4)]}})})}function $i(s){if(s<0||s>=256)throw new Error("Mask value must be in range [0, 255] but got "+s);if(!Number.isInteger(s))throw new Error("Mask value must be an integer but got "+s)}var Mn={runtime:"mediapipe",enableSmoothing:!0,enableSegmentation:!1,smoothSegmentation:!0,modelType:"full"},Jh=function(){function s(e){this.mask=e}return s.prototype.toCanvasImageSource=function(){return ie(this,void 0,void 0,function(){return oe(this,function(e){return[2,this.mask]})})},s.prototype.toImageData=function(){return ie(this,void 0,void 0,function(){return oe(this,function(e){return[2,zi(this.mask)]})})},s.prototype.toTensor=function(){return ie(this,void 0,void 0,function(){return oe(this,function(e){return[2,Qh(this.mask)]})})},s.prototype.getUnderlyingType=function(){return"canvasimagesource"},s}();function Zh(s){return $i(s),"person"}var ef=function(){function s(e){var a,o=this;switch(this.width=0,this.height=0,this.selfieMode=!1,this.poseSolution=new Yh.Pose({locateFile:function(i,t){return e.solutionPath?e.solutionPath.replace(/\/+$/,"")+"/"+i:t+"/"+i}}),e.modelType){case"lite":a=0;break;case"heavy":a=2;break;case"full":default:a=1}this.poseSolution.setOptions({modelComplexity:a,smoothLandmarks:e.enableSmoothing,enableSegmentation:e.enableSegmentation,smoothSegmentation:e.smoothSegmentation,selfieMode:this.selfieMode}),this.poseSolution.onResults(function(i){if(o.height=i.image.height,o.width=i.image.width,i.poseLandmarks==null)o.poses=[];else{var t=o.translateOutput(i.poseLandmarks,i.poseWorldLandmarks);i.segmentationMask&&(t.segmentation={maskValueToLabel:Zh,mask:new Jh(i.segmentationMask)}),o.poses=[t]}})}return s.prototype.translateOutput=function(e,a){var o=this,i={keypoints:e.map(function(t,u){return{x:t.x*o.width,y:t.y*o.height,z:t.z,score:t.visibility,name:Cn[u]}})};return a!=null&&(i.keypoints3D=a.map(function(t,u){return{x:t.x,y:t.y,z:t.z,score:t.visibility,name:Cn[u]}})),i},s.prototype.estimatePoses=function(e,a,o){return ie(this,void 0,void 0,function(){var i,t;return oe(this,function(u){switch(u.label){case 0:return a&&a.flipHorizontal&&a.flipHorizontal!==this.selfieMode&&(this.selfieMode=a.flipHorizontal,this.poseSolution.setOptions({selfieMode:this.selfieMode})),e instanceof gn?(t=ImageData.bind,[4,Hr(e)]):[3,2];case 1:return i=new(t.apply(ImageData,[void 0,u.sent(),e.shape[1],e.shape[0]])),[3,3];case 2:i=e,u.label=3;case 3:return e=i,[4,this.poseSolution.send({image:e},o)];case 4:return u.sent(),[2,this.poses]}})})},s.prototype.dispose=function(){this.poseSolution.close()},s.prototype.reset=function(){this.poseSolution.reset()},s.prototype.initialize=function(){return this.poseSolution.initialize()},s}();function tf(s){return ie(this,void 0,void 0,function(){var e,a;return oe(this,function(o){switch(o.label){case 0:return e=function(i){if(i==null)return de({},Mn);var t=de({},i);return t.runtime="mediapipe",t.enableSegmentation==null&&(t.enableSegmentation=Mn.enableSegmentation),t.enableSmoothing==null&&(t.enableSmoothing=Mn.enableSmoothing),t.smoothSegmentation==null&&(t.smoothSegmentation=Mn.smoothSegmentation),t.modelType==null&&(t.modelType=Mn.modelType),t}(s),[4,(a=new ef(e)).initialize()];case 1:return o.sent(),[2,a]}})})}function Nn(s){return s instanceof gn?{height:s.shape[0],width:s.shape[1]}:{height:s.height,width:s.width}}function Gi(s){return s-2*Math.PI*Math.floor((s+Math.PI)/(2*Math.PI))}function zr(s){return s instanceof gn?s:Os(s)}function qi(s,e,a){return Pr(a,"inputResolution"),[1/a.width*s[0][0]*e.width,1/a.height*s[0][1]*e.width,s[0][3]*e.width,1/a.width*s[1][0]*e.height,1/a.height*s[1][1]*e.height,s[1][3]*e.height,0,0]}function Pr(s,e){Be(s.width!==0,function(){return e+" width cannot be 0."}),Be(s.height!==0,function(){return e+" height cannot be 0."})}function vr(s,e,a){var o=a.rotationVectorStartKeypointIndex,i=a.rotationVectorEndKeypointIndex,t=s.locationData,u=t.relativeKeypoints[o].x*e.width,c=t.relativeKeypoints[o].y*e.height,p=t.relativeKeypoints[i].x*e.width,h=t.relativeKeypoints[i].y*e.height,r=2*Math.sqrt((p-u)*(p-u)+(h-c)*(h-c)),f=function(b,g,_){var v,x=b.locationData,S=_.rotationVectorStartKeypointIndex,k=_.rotationVectorEndKeypointIndex;v=_.rotationVectorTargetAngle?_.rotationVectorTargetAngle:Math.PI*_.rotationVectorTargetAngleDegree/180;var I=x.relativeKeypoints[S].x*g.width,R=x.relativeKeypoints[S].y*g.height,N=x.relativeKeypoints[k].x*g.width,E=x.relativeKeypoints[k].y*g.height;return Gi(v-Math.atan2(-(E-R),N-I))}(s,e,a);return{xCenter:u/e.width,yCenter:c/e.height,width:r/e.width,height:r/e.height,rotation:f}}function Ki(s){if(s.length!==16)throw new Error("Array length must be 16 but got "+s.length);return[[s[0],s[1],s[2],s[3]],[s[4],s[5],s[6],s[7]],[s[8],s[9],s[10],s[11]],[s[12],s[13],s[14],s[15]]]}function wr(s,e,a,o,i,t,u){return s[e][i]*(s[a][t]*s[o][u]-s[a][u]*s[o][t])}function Oe(s,e,a){var o=(e+1)%4,i=(e+2)%4,t=(e+3)%4,u=(a+1)%4,c=(a+2)%4,p=(a+3)%4;return wr(s,o,i,t,u,c,p)+wr(s,i,t,o,u,c,p)+wr(s,t,o,i,u,c,p)}function ta(s,e,a){a===void 0&&(a={ignoreRotation:!1});for(var o=[],i=0,t=s;i<t.length;i++){var u=t[i],c=u.x-.5,p=u.y-.5,h=a.ignoreRotation?0:e.rotation,r=Math.cos(h)*c-Math.sin(h)*p,f=Math.sin(h)*c+Math.cos(h)*p;r=r*e.width+e.xCenter,f=f*e.height+e.yCenter;var b=u.z*e.width,g=de({},u);g.x=r,g.y=f,g.z=b,o.push(g)}return o}function Xi(s,e){var a=function(o,i,t,u){var c=i-o,p=u-t;if(c===0)throw new Error("Original min and max are both "+o+", range cannot be 0.");var h=p/c;return{scale:h,offset:t-o*h}}(0,255,e[0],e[1]);return Pe(function(){return ze(Me(s,a.scale),a.offset)})}function Dr(s,e,a){var o,i,t,u,c,p,h,r,f,b,g,_,v,x,S=e.outputTensorSize,k=e.keepAspectRatio,I=e.borderMode,R=e.outputTensorFloatRange,N=Nn(s),E=function(B,D){return D?{xCenter:D.xCenter*B.width,yCenter:D.yCenter*B.height,width:D.width*B.width,height:D.height*B.height,rotation:D.rotation}:{xCenter:.5*B.width,yCenter:.5*B.height,width:B.width,height:B.height,rotation:0}}(N,a),O=function(B,D,V){if(V===void 0&&(V=!1),!V)return{top:0,left:0,right:0,bottom:0};var q=D.height,$=D.width;Pr(D,"targetSize"),Pr(B,"roi");var Y,Q,ee=q/$,ce=B.height/B.width,ae=0,xe=0;return ee>ce?(Y=B.width,Q=B.width*ee,xe=(1-ce/ee)/2):(Y=B.height/ee,Q=B.height,ae=(1-ee/ce)/2),B.width=Y,B.height=Q,{top:xe,left:ae,right:ae,bottom:xe}}(E,S,k),j=(o=E,i=N.width,t=N.height,u=!1,c=o.width,p=o.height,h=u?-1:1,r=Math.cos(o.rotation),f=Math.sin(o.rotation),b=o.xCenter,g=o.yCenter,_=1/i,v=1/t,(x=new Array(16))[0]=c*r*h*_,x[1]=-p*f*_,x[2]=0,x[3]=(-.5*c*r*h+.5*p*f+b)*_,x[4]=c*f*h*v,x[5]=p*r*v,x[6]=0,x[7]=(-.5*p*r-.5*c*f*h+g)*v,x[8]=0,x[9]=0,x[10]=c*_,x[11]=0,x[12]=0,x[13]=0,x[14]=0,x[15]=1,Ki(x));return{imageTensor:Pe(function(){var B=zr(s),D=qt(qi(j,N,S),[1,8]),V=I==="zero"?"constant":"nearest",q=Kt.transform(Rn(Fn(B,"float32")),D,"bilinear",V,0,[S.height,S.width]);return R!=null?Xi(q,R):q}),padding:O,transformationMatrix:j}}function na(s,e,a,o){return o===1?.5*(s+e):s+(e-s)*a/(o-1)}function nf(s){return Pe(function(){var e=function(i){return Pe(function(){return[Ve(i,[0,0,0],[1,-1,1]),Ve(i,[0,0,1],[1,-1,-1])]})}(s),a=e[0],o=e[1];return{boxes:me(o),logits:me(a)}})}function Yi(s){return s!=null&&s.currentTime!=null}function ra(s){for(var e={locationData:{relativeKeypoints:[]}},a=Number.MAX_SAFE_INTEGER,o=Number.MIN_SAFE_INTEGER,i=Number.MAX_SAFE_INTEGER,t=Number.MIN_SAFE_INTEGER,u=0;u<s.length;++u){var c=s[u];a=Math.min(a,c.x),o=Math.max(o,c.x),i=Math.min(i,c.y),t=Math.max(t,c.y),e.locationData.relativeKeypoints.push({x:c.x,y:c.y})}return e.locationData.relativeBoundingBox={xMin:a,yMin:i,xMax:o,yMax:t,width:o-a,height:t-i},e}function rf(s,e,a,o){return ie(this,void 0,void 0,function(){var i,t,u,c,p;return oe(this,function(h){switch(h.label){case 0:return s.sort(function(r,f){return Math.max.apply(Math,f.score)-Math.max.apply(Math,r.score)}),i=qt(s.map(function(r){return[r.locationData.relativeBoundingBox.yMin,r.locationData.relativeBoundingBox.xMin,r.locationData.relativeBoundingBox.yMax,r.locationData.relativeBoundingBox.xMax]})),t=An(s.map(function(r){return r.score[0]})),[4,Kt.nonMaxSuppressionAsync(i,t,e,a)];case 1:return[4,(u=h.sent()).array()];case 2:return c=h.sent(),p=s.filter(function(r,f){return c.indexOf(f)>-1}),De([i,t,u]),[2,p]}})})}function Qi(s,e){return s.map(function(a){var o=de(de({},a),{x:a.x*e.width,y:a.y*e.height});return a.z!=null&&(o.z=a.z*e.width),o})}function af(s,e,a){return ie(this,void 0,void 0,function(){var o,i,t,u,c,p,h,r,f,b,g,_,v,x,S,k,I,R,N,E,O,j,B,D;return oe(this,function(V){switch(V.label){case 0:if(o=me(e,[0]),i=o.shape,t=i[0],u=i[1],c=i[2],s.length!==c)throw new Error("Expected heatmap to have same number of channels as the number of landmarks. But got landmarks length: "+s.length+", heatmap length: "+c);return p=[],[4,o.buffer()];case 1:for(h=V.sent(),r=0;r<s.length;r++)if(f=s[r],b=de({},f),p.push(b),g=Math.trunc(b.x*u),_=Math.trunc(b.y*t),!(g<0||g>=u||_<0||g>=t)){for(v=Math.trunc((a.kernelSize-1)/2),x=Math.max(0,g-v),S=Math.min(u,g+v+1),k=Math.max(0,_-v),I=Math.min(t,_+v+1),R=0,N=0,E=0,O=0,j=k;j<I;++j)for(B=x;B<S;++B)D=h.get(j,B,r),R+=D,O=Math.max(O,D),N+=B*D,E+=j*D;O>=a.minConfidenceToRefine&&R>0&&(b.x=N/u/R,b.y=E/t/R)}return o.dispose(),[2,p]}})})}function aa(s,e){var a=e.left,o=e.top,i=e.left+e.right,t=e.top+e.bottom;return s.map(function(u){return de(de({},u),{x:(u.x-a)/(1-i),y:(u.y-o)/(1-t),z:u.z/(1-i)})})}function sf(s,e,a){return Zn()==="webgl"?function(o,i,t){var u=t.combineWithPreviousRatio.toFixed(2),c={variableNames:["prevMask","newMask"],outputShape:o.shape,userCode:`
  void main() {
      ivec2 coords = getOutputCoords();
      int height = coords[0];
      int width = coords[1];

      float prevMaskValue = getPrevMask(height, width);
      float newMaskValue = getNewMask(height, width);

      /*
      * Assume p := newMaskValue
      * H(p) := 1 + (p * log(p) + (1-p) * log(1-p)) / log(2)
      * uncertainty alpha(p) =
      *   Clamp(1 - (1 - H(p)) * (1 - H(p)), 0, 1) [squaring the
      * uncertainty]
      *
      * The following polynomial approximates uncertainty alpha as a
      * function of (p + 0.5):
      */
      const float c1 = 5.68842;
      const float c2 = -0.748699;
      const float c3 = -57.8051;
      const float c4 = 291.309;
      const float c5 = -624.717;
      float t = newMaskValue - 0.5;
      float x = t * t;

      float uncertainty =
        1.0 - min(1.0, x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5)))));

      float outputValue = newMaskValue + (prevMaskValue - newMaskValue) *
                             (uncertainty * `+u+`);

      setOutput(outputValue);
    }
`},p=qu();return Pe(function(){var h=p.compileAndRun(c,[o,i]);return Ns().makeTensorFromDataId(h.dataId,h.shape,h.dtype)})}(s,e,a):Pe(function(){var o=hn(e,.5),i=Ku(o),t=hn(1,Xu(1,Me(i,ze(5.68842,Me(i,ze(-.748699,Me(i,ze(-57.8051,Me(i,ze(291.309,Me(i,-624.717)))))))))));return ze(e,Me(hn(s,e),Me(t,a.combineWithPreviousRatio)))})}function of(s,e,a){return ie(this,void 0,void 0,function(){var o,i,t,u,c;return oe(this,function(p){switch(p.label){case 0:return o=s[0],i=s[1],t=function(h,r,f){return Pe(function(){var b,g,_,v;f.reverseOutputOrder?(g=me(Ve(h,[0,f.boxCoordOffset+0],[-1,1])),b=me(Ve(h,[0,f.boxCoordOffset+1],[-1,1])),v=me(Ve(h,[0,f.boxCoordOffset+2],[-1,1])),_=me(Ve(h,[0,f.boxCoordOffset+3],[-1,1]))):(b=me(Ve(h,[0,f.boxCoordOffset+0],[-1,1])),g=me(Ve(h,[0,f.boxCoordOffset+1],[-1,1])),_=me(Ve(h,[0,f.boxCoordOffset+2],[-1,1])),v=me(Ve(h,[0,f.boxCoordOffset+3],[-1,1]))),g=ze(Me(Ue(g,f.xScale),r.w),r.x),b=ze(Me(Ue(b,f.yScale),r.h),r.y),f.applyExponentialOnBoxSize?(_=Me(Yr(Ue(_,f.hScale)),r.h),v=Me(Yr(Ue(v,f.wScale)),r.w)):(_=Me(Ue(_,f.hScale),r.h),v=Me(Ue(v,f.wScale),r.h));var x=hn(b,Ue(_,2)),S=hn(g,Ue(v,2)),k=ze(b,Ue(_,2)),I=ze(g,Ue(v,2)),R=Fr([Rt(x,[f.numBoxes,1]),Rt(S,[f.numBoxes,1]),Rt(k,[f.numBoxes,1]),Rt(I,[f.numBoxes,1])],1);if(f.numKeypoints)for(var N=0;N<f.numKeypoints;++N){var E=f.keypointCoordOffset+N*f.numValuesPerKeypoint,O=void 0,j=void 0;f.reverseOutputOrder?(O=me(Ve(h,[0,E],[-1,1])),j=me(Ve(h,[0,E+1],[-1,1]))):(j=me(Ve(h,[0,E],[-1,1])),O=me(Ve(h,[0,E+1],[-1,1])));var B=ze(Me(Ue(O,f.xScale),r.w),r.x),D=ze(Me(Ue(j,f.yScale),r.h),r.y);R=Fr([R,Rt(B,[f.numBoxes,1]),Rt(D,[f.numBoxes,1])],1)}return R})}(i,e,a),u=Pe(function(){var h=o;return a.sigmoidScore?(a.scoreClippingThresh!=null&&(h=Yu(o,-a.scoreClippingThresh,a.scoreClippingThresh)),h=Vr(h)):h}),[4,uf(t,u,a)];case 1:return c=p.sent(),De([t,u]),[2,c]}})})}function uf(s,e,a){return ie(this,void 0,void 0,function(){var o,i,t,u,c,p,h,r,f,b,g,_;return oe(this,function(v){switch(v.label){case 0:return o=[],[4,s.data()];case 1:return i=v.sent(),[4,e.data()];case 2:for(t=v.sent(),u=0;u<a.numBoxes;++u)if(!(a.minScoreThresh!=null&&t[u]<a.minScoreThresh||(c=u*a.numCoords,p=lf(i[c+0],i[c+1],i[c+2],i[c+3],t[u],a.flipVertically,u),(h=p.locationData.relativeBoundingBox).width<0||h.height<0))){if(a.numKeypoints>0)for((r=p.locationData).relativeKeypoints=[],f=a.numKeypoints*a.numValuesPerKeypoint,b=0;b<f;b+=a.numValuesPerKeypoint)g=c+a.keypointCoordOffset+b,_={x:i[g+0],y:a.flipVertically?1-i[g+1]:i[g+1]},r.relativeKeypoints.push(_);o.push(p)}return[2,o]}})})}function lf(s,e,a,o,i,t,u){return{score:[i],ind:u,locationData:{relativeBoundingBox:{xMin:e,yMin:t?1-a:s,xMax:o,yMax:t?1-s:a,width:o-e,height:a-s}}}}function cf(s,e){return s==="none"?e:function(a){return 1/(1+Math.exp(-a))}(e)}function sa(s,e,a,o){return ie(this,void 0,void 0,function(){var i,t,u,c,p,h,r,f;return oe(this,function(b){switch(b.label){case 0:return a=a||e.flipHorizontally||!1,o=o||e.flipVertically||!1,i=s.size,t=i/e.numLandmarks,[4,s.data()];case 1:for(u=b.sent(),c=[],p=0;p<e.numLandmarks;++p)h=p*t,(f={x:0,y:0}).x=a?e.inputImageWidth-u[h]:u[h],t>1&&(f.y=o?e.inputImageHeight-u[h+1]:u[h+1]),t>2&&(f.z=u[h+2]),t>3&&(f.score=cf(e.visibilityActivation,u[h+3])),c.push(f);for(r=0;r<c.length;++r)(f=c[r]).x=f.x/e.inputImageWidth,f.y=f.y/e.inputImageHeight,f.z=f.z/e.inputImageWidth/(e.normalizeZ||1);return[2,c]}})})}function ia(s,e,a){var o=s.width,i=s.height,t=s.rotation;if(a.rotation==null&&a.rotationDegree==null||(t=function(r,f){return f.rotation!=null?r+=f.rotation:f.rotationDegree!=null&&(r+=Math.PI*f.rotationDegree/180),Gi(r)}(t,a)),t===0)s.xCenter=s.xCenter+o*a.shiftX,s.yCenter=s.yCenter+i*a.shiftY;else{var u=(e.width*o*a.shiftX*Math.cos(t)-e.height*i*a.shiftY*Math.sin(t))/e.width,c=(e.width*o*a.shiftX*Math.sin(t)+e.height*i*a.shiftY*Math.cos(t))/e.height;s.xCenter=s.xCenter+u,s.yCenter=s.yCenter+c}if(a.squareLong){var p=Math.max(o*e.width,i*e.height);o=p/e.width,i=p/e.height}else if(a.squareShort){var h=Math.min(o*e.width,i*e.height);o=h/e.width,i=h/e.height}return s.width=o*a.scaleX,s.height=i*a.scaleY,s}function pf(s,e){return s.map(function(a){var o=de(de({},a),{x:a.x/e.width,y:a.y/e.height});return a.z!=null&&(a.z=a.z/e.width),o})}var Ft=function(){function s(e){this.alpha=e,this.initialized=!1}return s.prototype.apply=function(e,a){var o;return this.initialized?o=a==null?this.storedValue+this.alpha*(e-this.storedValue):this.storedValue+this.alpha*a*Math.asinh((e-this.storedValue)/a):(o=e,this.initialized=!0),this.rawValue=e,this.storedValue=o,o},s.prototype.applyWithAlpha=function(e,a,o){return this.alpha=a,this.apply(e,o)},s.prototype.hasLastRawValue=function(){return this.initialized},s.prototype.lastRawValue=function(){return this.rawValue},s.prototype.reset=function(){this.initialized=!1},s}(),kr=function(){function s(e){this.frequency=e.frequency,this.minCutOff=e.minCutOff,this.beta=e.beta,this.thresholdCutOff=e.thresholdCutOff,this.thresholdBeta=e.thresholdBeta,this.derivateCutOff=e.derivateCutOff,this.x=new Ft(this.getAlpha(this.minCutOff)),this.dx=new Ft(this.getAlpha(this.derivateCutOff)),this.lastTimestamp=0}return s.prototype.apply=function(e,a,o){if(e==null)return e;var i=Math.trunc(a);if(this.lastTimestamp>=i)return e;this.lastTimestamp!==0&&i!==0&&(this.frequency=1/(1e-6*(i-this.lastTimestamp))),this.lastTimestamp=i;var t=this.x.hasLastRawValue()?(e-this.x.lastRawValue())*o*this.frequency:0,u=this.dx.applyWithAlpha(t,this.getAlpha(this.derivateCutOff)),c=this.minCutOff+this.beta*Math.abs(u),p=this.thresholdCutOff!=null?this.thresholdCutOff+this.thresholdBeta*Math.abs(u):null;return this.x.applyWithAlpha(e,this.getAlpha(c),p)},s.prototype.getAlpha=function(e){return 1/(1+this.frequency/(2*Math.PI*e))},s}(),Br=function(){function s(e){this.config=e}return s.prototype.apply=function(e,a,o){var i=this;if(e==null)return this.reset(),null;this.initializeFiltersIfEmpty(e);var t=1;if(!this.config.disableValueScaling){if(o<this.config.minAllowedObjectScale)return Gt(e);t=1/o}return e.map(function(u,c){var p=de(de({},u),{x:i.xFilters[c].apply(u.x,a,t),y:i.yFilters[c].apply(u.y,a,t)});return u.z!=null&&(p.z=i.zFilters[c].apply(u.z,a,t)),p})},s.prototype.reset=function(){this.xFilters=null,this.yFilters=null,this.zFilters=null},s.prototype.initializeFiltersIfEmpty=function(e){var a=this;this.xFilters!=null&&this.xFilters.length===e.length||(this.xFilters=e.map(function(o){return new kr(a.config)}),this.yFilters=e.map(function(o){return new kr(a.config)}),this.zFilters=e.map(function(o){return new kr(a.config)}))},s}(),Ir=function(){function s(e){this.config=e,this.window=[],this.lowPassFilter=new Ft(1),this.lastValue=0,this.lastValueScale=1,this.lastTimestamp=-1}return s.prototype.apply=function(e,a,o){if(e==null)return e;var i,t=Math.trunc(a);if(this.lastTimestamp>=t)return e;if(this.lastTimestamp===-1)i=1;else{for(var u=e*o-this.lastValue*this.lastValueScale,c=t-this.lastTimestamp,p=u,h=c,r=(1+this.window.length)*(1e6/30),f=0,b=this.window;f<b.length;f++){var g=b[f];if(h+g.duration>r)break;p+=g.distance,h+=g.duration}var _=p/(1e-6*h);i=1-1/(1+this.config.velocityScale*Math.abs(_)),this.window.unshift({distance:u,duration:c}),this.window.length>this.config.windowSize&&this.window.pop()}return this.lastValue=e,this.lastValueScale=o,this.lastTimestamp=t,this.lowPassFilter.applyWithAlpha(e,i)},s}(),df=function(){function s(e){this.config=e}return s.prototype.apply=function(e,a,o){var i=this;if(e==null)return this.reset(),null;var t=1;if(!this.config.disableValueScaling){if(o<this.config.minAllowedObjectScale)return Gt(e);t=1/o}return this.initializeFiltersIfEmpty(e),e.map(function(u,c){var p=de(de({},u),{x:i.xFilters[c].apply(u.x,a,t),y:i.yFilters[c].apply(u.y,a,t)});return u.z!=null&&(p.z=i.zFilters[c].apply(u.z,a,t)),p})},s.prototype.reset=function(){this.xFilters=null,this.yFilters=null,this.zFilters=null},s.prototype.initializeFiltersIfEmpty=function(e){var a=this;this.xFilters!=null&&this.xFilters.length===e.length||(this.xFilters=e.map(function(o){return new Ir(a.config)}),this.yFilters=e.map(function(o){return new Ir(a.config)}),this.zFilters=e.map(function(o){return new Ir(a.config)}))},s}(),xr=function(){function s(e){if(e.velocityFilter!=null)this.keypointsFilter=new df(e.velocityFilter);else{if(e.oneEuroFilter==null)throw new Error("Either configure velocityFilter or oneEuroFilter, but got "+e+".");this.keypointsFilter=new Br(e.oneEuroFilter)}}return s.prototype.apply=function(e,a,o,i,t){if(i===void 0&&(i=!1),e==null)return this.keypointsFilter.reset(),null;var u=t!=null?function(h,r){return(h.width*r.width+h.height*r.height)/2}(t,o):1,c=i?Qi(e,o):e,p=this.keypointsFilter.apply(c,a,u);return i?pf(p,o):p},s}(),oa=function(){function s(e){this.alpha=e.alpha}return s.prototype.apply=function(e){var a=this;if(e==null)return this.visibilityFilters=null,null;this.visibilityFilters!=null&&this.visibilityFilters.length===e.length||(this.visibilityFilters=e.map(function(c){return new Ft(a.alpha)}));for(var o=[],i=0;i<e.length;++i){var t=e[i],u=de({},t);u.score=this.visibilityFilters[i].apply(t.score),o.push(u)}return o},s}(),hf={reduceBoxesInLowestlayer:!1,interpolatedScaleAspectRatio:1,featureMapHeight:[],featureMapWidth:[],numLayers:5,minScale:.1484375,maxScale:.75,inputSizeHeight:224,inputSizeWidth:224,anchorOffsetX:.5,anchorOffsetY:.5,strides:[8,16,32,32,32],aspectRatios:[1],fixedAnchorSize:!0},dn={runtime:"tfjs",modelType:"full",enableSmoothing:!0,enableSegmentation:!1,smoothSegmentation:!0,detectorModelUrl:"https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/detector/1",landmarkModelUrl:"https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/landmark/full/2"},ff={maxPoses:1,flipHorizontal:!1},mf={applyExponentialOnBoxSize:!1,flipVertically:!1,ignoreClasses:[],numClasses:1,numBoxes:2254,numCoords:12,boxCoordOffset:0,keypointCoordOffset:4,numKeypoints:4,numValuesPerKeypoint:2,sigmoidScore:!0,scoreClippingThresh:100,reverseOutputOrder:!0,xScale:224,yScale:224,hScale:224,wScale:224,minScoreThresh:.5},gf=.3,ua={shiftX:0,shiftY:0,scaleX:1.25,scaleY:1.25,squareLong:!0},yf={outputTensorSize:{width:224,height:224},keepAspectRatio:!0,outputTensorFloatRange:[-1,1],borderMode:"zero"},bf={outputTensorSize:{width:256,height:256},keepAspectRatio:!0,outputTensorFloatRange:[0,1],borderMode:"zero"},_f={numLandmarks:39,inputImageWidth:256,inputImageHeight:256,visibilityActivation:"sigmoid",flipHorizontally:!1,flipVertically:!1},vf={numLandmarks:39,inputImageWidth:1,inputImageHeight:1,visibilityActivation:"sigmoid",flipHorizontally:!1,flipVertically:!1},wf={kernelSize:7,minConfidenceToRefine:.5},la={alpha:.1},kf={oneEuroFilter:{frequency:30,minCutOff:.05,beta:80,derivateCutOff:1,minAllowedObjectScale:1e-6}},If={oneEuroFilter:{frequency:30,minCutOff:.01,beta:10,derivateCutOff:1,minAllowedObjectScale:1e-6}},xf={oneEuroFilter:{frequency:30,minCutOff:.1,beta:40,derivateCutOff:1,minAllowedObjectScale:1e-6,disableValueScaling:!0}},Sf={activation:"none"},Mf={combineWithPreviousRatio:.7},Af=function(){function s(e){this.mask=e}return s.prototype.toCanvasImageSource=function(){return ie(this,void 0,void 0,function(){return oe(this,function(e){return[2,Ui(this.mask)]})})},s.prototype.toImageData=function(){return ie(this,void 0,void 0,function(){return oe(this,function(e){return[2,zi(this.mask)]})})},s.prototype.toTensor=function(){return ie(this,void 0,void 0,function(){return oe(this,function(e){return[2,this.mask]})})},s.prototype.getUnderlyingType=function(){return"tensor"},s}();function Tf(s){return $i(s),"person"}var Rf=function(){function s(e,a,o,i,t,u){this.detectorModel=e,this.landmarkModel=a,this.enableSmoothing=o,this.enableSegmentation=i,this.smoothSegmentation=t,this.modelType=u,this.regionOfInterest=null,this.prevFilteredSegmentationMask=null,this.anchors=function(f){f.reduceBoxesInLowestLayer==null&&(f.reduceBoxesInLowestLayer=!1),f.interpolatedScaleAspectRatio==null&&(f.interpolatedScaleAspectRatio=1),f.fixedAnchorSize==null&&(f.fixedAnchorSize=!1);for(var b=[],g=0;g<f.numLayers;){for(var _=[],v=[],x=[],S=[],k=g;k<f.strides.length&&f.strides[k]===f.strides[g];){var I=na(f.minScale,f.maxScale,k,f.strides.length);if(k===0&&f.reduceBoxesInLowestLayer)x.push(1),x.push(2),x.push(.5),S.push(.1),S.push(I),S.push(I);else{for(var R=0;R<f.aspectRatios.length;++R)x.push(f.aspectRatios[R]),S.push(I);if(f.interpolatedScaleAspectRatio>0){var N=k===f.strides.length-1?1:na(f.minScale,f.maxScale,k+1,f.strides.length);S.push(Math.sqrt(I*N)),x.push(f.interpolatedScaleAspectRatio)}}k++}for(var E=0;E<x.length;++E){var O=Math.sqrt(x[E]);_.push(S[E]/O),v.push(S[E]*O)}var j=0,B=0;if(f.featureMapHeight.length>0)j=f.featureMapHeight[g],B=f.featureMapWidth[g];else{var D=f.strides[g];j=Math.ceil(f.inputSizeHeight/D),B=Math.ceil(f.inputSizeWidth/D)}for(var V=0;V<j;++V)for(var q=0;q<B;++q)for(var $=0;$<_.length;++$){var Y={xCenter:(q+f.anchorOffsetX)/B,yCenter:(V+f.anchorOffsetY)/j,width:0,height:0};f.fixedAnchorSize?(Y.width=1,Y.height=1):(Y.width=v[$],Y.height=_[$]),b.push(Y)}g=k}return b}(hf);var c=An(this.anchors.map(function(f){return f.width})),p=An(this.anchors.map(function(f){return f.height})),h=An(this.anchors.map(function(f){return f.xCenter})),r=An(this.anchors.map(function(f){return f.yCenter}));this.anchorTensor={x:h,y:r,w:c,h:p},this.prevFilteredSegmentationMask=this.enableSegmentation?qt([],[0,0]):null}return s.prototype.estimatePoses=function(e,a,o){return ie(this,void 0,void 0,function(){var i,t,u,c,p,h,r,f,b,g,_,v,x,S,k,I,R,N,E,O,j,B,D;return oe(this,function(V){switch(V.label){case 0:return i=function(q){var $;if(($=q==null?ff:de({},q)).maxPoses==null&&($.maxPoses=1),$.maxPoses<=0)throw new Error("Invalid maxPoses "+$.maxPoses+". Should be > 0.");if($.maxPoses>1)throw new Error("Multi-pose detection is not implemented yet. Please set maxPoses to 1.");return $}(a),e==null?(this.reset(),[2,[]]):(this.maxPoses=i.maxPoses,this.timestamp=o!=null?1e3*o:Yi(e)?1e6*e.currentTime:null,t=Nn(e),u=Pe(function(){return Fn(zr(e),"float32")}),(c=this.regionOfInterest)!=null?[3,2]:[4,this.detectPose(u)]);case 1:if((p=V.sent()).length===0)return this.reset(),u.dispose(),[2,[]];h=p[0],c=this.poseDetectionToRoi(h,t),V.label=2;case 2:return[4,this.poseLandmarksByRoi(c,u)];case 3:return r=V.sent(),u.dispose(),r==null?(this.reset(),[2,[]]):(f=r.landmarks,b=r.auxiliaryLandmarks,g=r.poseScore,_=r.worldLandmarks,v=r.segmentationMask,x=this.poseLandmarkFiltering(f,b,_,t),S=x.actualLandmarksFiltered,k=x.auxiliaryLandmarksFiltered,I=x.actualWorldLandmarksFiltered,R=this.poseLandmarksToRoi(k,t),this.regionOfInterest=R,N=this.smoothSegmentation&&v!=null?this.poseSegmentationFiltering(v):v,(E=S!=null?Qi(S,t):null)!=null&&E.forEach(function(q,$){q.name=Cn[$]}),(O=I)!=null&&O.forEach(function(q,$){q.name=Cn[$]}),j={score:g,keypoints:E,keypoints3D:O},N!==null&&(B=Pe(function(){var q=Rn(N,2),$=Rr(q,[[0,0],[0,0],[0,1]]);return $u($,[[0,0],[0,0],[0,2]],"symmetric")}),this.smoothSegmentation||De(N),D={maskValueToLabel:Tf,mask:new Af(B)},j.segmentation=D),[2,[j]])}})})},s.prototype.poseSegmentationFiltering=function(e){var a=this.prevFilteredSegmentationMask;return a.size===0?this.prevFilteredSegmentationMask=e:(this.prevFilteredSegmentationMask=sf(a,e,Mf),De(e)),De(a),this.prevFilteredSegmentationMask},s.prototype.dispose=function(){this.detectorModel.dispose(),this.landmarkModel.dispose(),De([this.anchorTensor.x,this.anchorTensor.y,this.anchorTensor.w,this.anchorTensor.h,this.prevFilteredSegmentationMask])},s.prototype.reset=function(){this.regionOfInterest=null,this.enableSegmentation&&(De(this.prevFilteredSegmentationMask),this.prevFilteredSegmentationMask=qt([],[0,0])),this.visibilitySmoothingFilterActual=null,this.visibilitySmoothingFilterAuxiliary=null,this.landmarksSmoothingFilterActual=null,this.landmarksSmoothingFilterAuxiliary=null},s.prototype.detectPose=function(e){return ie(this,void 0,void 0,function(){var a,o,i,t,u,c,p,h,r,f;return oe(this,function(b){switch(b.label){case 0:return a=Dr(e,yf),o=a.imageTensor,i=a.padding,t=this.detectorModel.predict(o),u=nf(t),c=u.boxes,[4,of([p=u.logits,c],this.anchorTensor,mf)];case 1:return(h=b.sent()).length===0?(De([o,t,p,c]),[2,h]):[4,rf(h,this.maxPoses,gf)];case 2:return r=b.sent(),f=function(g,_){g===void 0&&(g=[]);for(var v=_.left,x=_.top,S=_.left+_.right,k=_.top+_.bottom,I=0;I<g.length;I++){var R=g[I],N=R.locationData.relativeBoundingBox,E=(N.xMin-v)/(1-S),O=(N.yMin-x)/(1-k),j=N.width/(1-S),B=N.height/(1-k);N.xMin=E,N.yMin=O,N.width=j,N.height=B,N.xMax=E+j,N.yMax=O+B;var D=R.locationData.relativeKeypoints;D&&D.forEach(function(V){var q=(V.x-v)/(1-S),$=(V.y-x)/(1-k);V.x=q,V.y=$})}return g}(r,i),De([o,t,p,c]),[2,f]}})})},s.prototype.poseDetectionToRoi=function(e,a){return ia(vr(e,a,{rotationVectorEndKeypointIndex:1,rotationVectorStartKeypointIndex:0,rotationVectorTargetAngleDegree:90}),a,ua)},s.prototype.poseLandmarksByRoi=function(e,a){return ie(this,void 0,void 0,function(){var o,i,t,u,c,p,h,r,f,b,g,_,v,x;return oe(this,function(S){switch(S.label){case 0:if(o=Nn(a),i=Dr(a,bf,e),t=i.imageTensor,u=i.padding,c=i.transformationMatrix,this.modelType!=="lite"&&this.modelType!=="full"&&this.modelType!=="heavy")throw new Error("Model type must be one of lite, full or heavy,but got "+this.modelType);return p=["ld_3d","output_poseflag","activation_heatmap","world_3d"],this.enableSegmentation&&p.push("activation_segmentation"),h=this.landmarkModel.execute(t,p),[4,this.tensorsToPoseLandmarksAndSegmentation(h)];case 1:return(r=S.sent())==null?(De(h),De(t),[2,null]):(f=r.landmarks,b=r.auxiliaryLandmarks,g=r.poseScore,_=r.worldLandmarks,v=r.segmentationMask,[4,this.poseLandmarksAndSegmentationInverseProjection(o,e,u,c,f,b,_,v)]);case 2:return x=S.sent(),De(h),De(t),[2,de({poseScore:g},x)]}})})},s.prototype.poseLandmarksAndSegmentationInverseProjection=function(e,a,o,i,t,u,c,p){return ie(this,void 0,void 0,function(){var h,r,f,b,g,_;return oe(this,function(v){return h=aa(t,o),r=aa(u,o),f=ta(h,a),b=ta(r,a),g=function(x,S){for(var k=[],I=0,R=x;I<R.length;I++){var N=R[I],E=N.x,O=N.y,j=S.rotation,B=Math.cos(j)*E-Math.sin(j)*O,D=Math.sin(j)*E+Math.cos(j)*O,V=de({},N);V.x=B,V.y=D,k.push(V)}return k}(c,a),_=null,this.enableSegmentation&&(_=Pe(function(){var x=p.shape,S=x[0],k=x[1],I=function(E){var O=Ki(new Array(16).fill(0));O[0][0]=Oe(E,0,0),O[1][0]=-Oe(E,0,1),O[2][0]=Oe(E,0,2),O[3][0]=-Oe(E,0,3),O[0][2]=Oe(E,2,0),O[1][2]=-Oe(E,2,1),O[2][2]=Oe(E,2,2),O[3][2]=-Oe(E,2,3),O[0][1]=-Oe(E,1,0),O[1][1]=Oe(E,1,1),O[2][1]=-Oe(E,1,2),O[3][1]=Oe(E,1,3),O[0][3]=-Oe(E,3,0),O[1][3]=Oe(E,3,1),O[2][3]=-Oe(E,3,2),O[3][3]=Oe(E,3,3);for(var j=E[0][0]*O[0][0]+E[1][0]*O[0][1]+E[2][0]*O[0][2]+E[3][0]*O[0][3],B=0;B<O.length;B++)for(var D=0;D<O.length;D++)O[B][D]/=j;return O}(i),R=qt(qi(I,{width:k,height:S},e),[1,8]),N=[1,S,k,1];return me(Kt.transform(Rt(p,N),R,"bilinear","constant",0,[e.height,e.width]),[0,3])}),De(p)),[2,{landmarks:f,auxiliaryLandmarks:b,worldLandmarks:g,segmentationMask:_}]})})},s.prototype.tensorsToPoseLandmarksAndSegmentation=function(e){return ie(this,void 0,void 0,function(){var a,o,i,t,u,c,p,h,r,f,b,g,_;return oe(this,function(v){switch(v.label){case 0:return a=e[0],o=e[1],i=e[2],t=e[3],u=this.enableSegmentation?e[4]:null,[4,o.data()];case 1:return(c=v.sent()[0])<.5?[2,null]:[4,sa(a,_f)];case 2:return[4,af(v.sent(),i,wf)];case 3:return p=v.sent(),h=p.slice(0,33),r=p.slice(33,35),[4,sa(t,vf)];case 4:return f=v.sent(),b=f.slice(0,33),g=function(x,S,k){k===void 0&&(k=!0);for(var I=[],R=0;R<x.length;R++){var N=de({},S[R]);k&&(N.score=x[R].score),I.push(N)}return I}(h,b,!0),_=this.enableSegmentation?function(x,S,k){return Pe(function(){var I=me(x,[0]),R=I.shape[2];if(R===1){var N=I;switch(S.activation){case"none":break;case"sigmoid":N=Vr(N);break;case"softmax":throw new Error("Softmax activation requires two channels.");default:throw new Error("Activation not supported ("+S.activation+")")}var E=k?Kt.resizeBilinear(N,[k.height,k.width]):N;return me(E,[2])}throw new Error("Unsupported number of tensor channels "+R)})}(u,Sf):null,[2,{landmarks:h,auxiliaryLandmarks:r,poseScore:c,worldLandmarks:g,segmentationMask:_}]}})})},s.prototype.poseLandmarksToRoi=function(e,a){return ia(vr(ra(e),a,{rotationVectorStartKeypointIndex:0,rotationVectorEndKeypointIndex:1,rotationVectorTargetAngleDegree:90}),a,ua)},s.prototype.poseLandmarkFiltering=function(e,a,o,i){var t,u,c;if(this.timestamp!=null&&this.enableSmoothing){var p=vr(ra(a),i,{rotationVectorEndKeypointIndex:0,rotationVectorStartKeypointIndex:1,rotationVectorTargetAngleDegree:90});this.visibilitySmoothingFilterActual==null&&(this.visibilitySmoothingFilterActual=new oa(la)),t=this.visibilitySmoothingFilterActual.apply(e),this.visibilitySmoothingFilterAuxiliary==null&&(this.visibilitySmoothingFilterAuxiliary=new oa(la)),u=this.visibilitySmoothingFilterAuxiliary.apply(a),c=this.visibilitySmoothingFilterActual.apply(o),this.landmarksSmoothingFilterActual==null&&(this.landmarksSmoothingFilterActual=new xr(kf)),t=this.landmarksSmoothingFilterActual.apply(t,this.timestamp,i,!0,p),this.landmarksSmoothingFilterAuxiliary==null&&(this.landmarksSmoothingFilterAuxiliary=new xr(If)),u=this.landmarksSmoothingFilterAuxiliary.apply(u,this.timestamp,i,!0,p),this.worldLandmarksSmoothingFilterActual==null&&(this.worldLandmarksSmoothingFilterActual=new xr(xf)),c=this.worldLandmarksSmoothingFilterActual.apply(o,this.timestamp)}else t=e,u=a,c=o;return{actualLandmarksFiltered:t,auxiliaryLandmarksFiltered:u,actualWorldLandmarksFiltered:c}},s}();function Ff(s){return ie(this,void 0,void 0,function(){var e,a,o,i,t,u;return oe(this,function(c){switch(c.label){case 0:return e=function(p){var h=de({},p??dn);if(h.enableSmoothing==null&&(h.enableSmoothing=dn.enableSmoothing),h.enableSegmentation==null&&(h.enableSegmentation=dn.enableSegmentation),h.smoothSegmentation==null&&(h.smoothSegmentation=dn.smoothSegmentation),h.modelType==null&&(h.modelType=dn.modelType),h.detectorModelUrl==null&&(h.detectorModelUrl=dn.detectorModelUrl),h.landmarkModelUrl==null)switch(h.modelType){case"lite":h.landmarkModelUrl="https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/landmark/lite/2";break;case"heavy":h.landmarkModelUrl="https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/landmark/heavy/2";break;case"full":default:h.landmarkModelUrl="https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/landmark/full/2"}return h}(s),a=typeof e.detectorModelUrl=="string"&&e.detectorModelUrl.indexOf("https://tfhub.dev")>-1,o=typeof e.landmarkModelUrl=="string"&&e.landmarkModelUrl.indexOf("https://tfhub.dev")>-1,[4,Promise.all([fn(e.detectorModelUrl,{fromTFHub:a}),fn(e.landmarkModelUrl,{fromTFHub:o})])];case 1:return i=c.sent(),t=i[0],u=i[1],[2,new Rf(t,u,e.enableSmoothing,e.enableSegmentation,e.smoothSegmentation,e.modelType)]}})})}var Xt,mt,Ji=function(){function s(e){(function(a){if(a.maxTracks<1)throw new Error("Must specify 'maxTracks' to be at least 1, but encountered "+a.maxTracks);if(a.maxAge<=0)throw new Error("Must specify 'maxAge' to be positive, but encountered "+a.maxAge);if(a.keypointTrackerParams!==void 0){if(a.keypointTrackerParams.keypointConfidenceThreshold<0||a.keypointTrackerParams.keypointConfidenceThreshold>1)throw new Error("Must specify 'keypointConfidenceThreshold' to be in the range [0, 1], but encountered "+a.keypointTrackerParams.keypointConfidenceThreshold);if(a.keypointTrackerParams.minNumberOfKeypoints<1)throw new Error("Must specify 'minNumberOfKeypoints' to be at least 1, but encountered "+a.keypointTrackerParams.minNumberOfKeypoints);for(var o=0,i=a.keypointTrackerParams.keypointFalloff;o<i.length;o++){var t=i[o];if(t<=0)throw new Error("Must specify each keypoint falloff parameterto be positive but encountered "+t)}}})(e),this.tracks=[],this.maxTracks=e.maxTracks,this.maxAge=1e3*e.maxAge,this.minSimilarity=e.minSimilarity,this.nextID=1}return s.prototype.apply=function(e,a){this.filterOldTracks(a);var o=this.computeSimilarity(e);return this.assignTracks(e,o,a),this.updateTracks(a),e},s.prototype.getTracks=function(){return this.tracks.slice()},s.prototype.getTrackIDs=function(){return new Set(this.tracks.map(function(e){return e.id}))},s.prototype.filterOldTracks=function(e){var a=this;this.tracks=this.tracks.filter(function(o){return e-o.lastTimestamp<=a.maxAge})},s.prototype.assignTracks=function(e,a,o){for(var i=Array.from(Array(a[0].length).keys()),t=[],u=0,c=Array.from(Array(e.length).keys());u<c.length;u++){var p=c[u];if(i.length!==0){for(var h=-1,r=-1,f=0,b=i;f<b.length;f++){var g=b[f],_=a[p][g];_>=this.minSimilarity&&_>r&&(h=g,r=_)}if(h>=0){var v=this.tracks[h];v=Object.assign(v,this.createTrack(e[p],o,v.id)),e[p].id=v.id;var x=i.indexOf(h);i.splice(x,1)}else t.push(p)}else t.push(p)}for(var S=0,k=t;S<k.length;S++){p=k[S];var I=this.createTrack(e[p],o);this.tracks.push(I),e[p].id=I.id}},s.prototype.updateTracks=function(e){this.tracks.sort(function(a,o){return o.lastTimestamp-a.lastTimestamp}),this.tracks=this.tracks.slice(0,this.maxTracks)},s.prototype.createTrack=function(e,a,o){var i={id:o||this.nextTrackID(),lastTimestamp:a,keypoints:Gt(e.keypoints).map(function(t){return de({},t)})};return e.box!==void 0&&(i.box=de({},e.box)),i},s.prototype.nextTrackID=function(){var e=this.nextID;return this.nextID+=1,e},s.prototype.remove=function(){for(var e=[],a=0;a<arguments.length;a++)e[a]=arguments[a];this.tracks=this.tracks.filter(function(o){return!e.includes(o.id)})},s.prototype.reset=function(){this.tracks=[]},s}(),Ef=function(s){function e(a){return s.call(this,a)||this}return Vi(e,s),e.prototype.computeSimilarity=function(a){var o=this;return a.length===0||this.tracks.length===0?[[]]:a.map(function(i){return o.tracks.map(function(t){return o.iou(i,t)})})},e.prototype.iou=function(a,o){var i=Math.max(a.box.xMin,o.box.xMin),t=Math.max(a.box.yMin,o.box.yMin),u=Math.min(a.box.xMax,o.box.xMax),c=Math.min(a.box.yMax,o.box.yMax);if(i>=u||t>=c)return 0;var p=(u-i)*(c-t);return p/(a.box.width*a.box.height+o.box.width*o.box.height-p)},e}(Ji),Cf=function(s){function e(a){var o=s.call(this,a)||this;return o.keypointThreshold=a.keypointTrackerParams.keypointConfidenceThreshold,o.keypointFalloff=a.keypointTrackerParams.keypointFalloff,o.minNumKeyoints=a.keypointTrackerParams.minNumberOfKeypoints,o}return Vi(e,s),e.prototype.computeSimilarity=function(a){if(a.length===0||this.tracks.length===0)return[[]];for(var o=[],i=0,t=a;i<t.length;i++){for(var u=t[i],c=[],p=0,h=this.tracks;p<h.length;p++){var r=h[p];c.push(this.oks(u,r))}o.push(c)}return o},e.prototype.oks=function(a,o){for(var i=this.area(o.keypoints)+1e-6,t=0,u=0,c=0;c<a.keypoints.length;++c){var p=a.keypoints[c],h=o.keypoints[c];if(!(p.score<this.keypointThreshold||h.score<this.keypointThreshold)){u+=1;var r=Math.pow(p.x-h.x,2)+Math.pow(p.y-h.y,2),f=2*this.keypointFalloff[c];t+=Math.exp(-1*r/(2*i*Math.pow(f,2)))}}return u<this.minNumKeyoints?0:t/u},e.prototype.area=function(a){var o=this,i=a.filter(function(p){return p.score>o.keypointThreshold}),t=Math.min.apply(Math,Gt([1],i.map(function(p){return p.x}))),u=Math.max.apply(Math,Gt([0],i.map(function(p){return p.x}))),c=Math.min.apply(Math,Gt([1],i.map(function(p){return p.y})));return(u-t)*(Math.max.apply(Math,Gt([0],i.map(function(p){return p.y})))-c)},e}(Ji);function Nf(s){switch(s){case mt.BlazePose:return Cn.reduce(function(e,a,o){return e[a]=o,e},{});case mt.PoseNet:case mt.MoveNet:return gt.reduce(function(e,a,o){return e[a]=o,e},{});default:throw new Error("Model "+s+" is not supported.")}}(function(s){s.Keypoint="keypoint",s.BoundingBox="boundingBox"})(Xt||(Xt={})),function(s){s.MoveNet="MoveNet",s.BlazePose="BlazePose",s.PoseNet="PoseNet"}(mt||(mt={}));var ca=["SinglePose.Lightning","SinglePose.Thunder","MultiPose.Lightning"],Zi={modelType:"SinglePose.Lightning",enableSmoothing:!0},pa={},da={frequency:30,minCutOff:2.5,beta:300,derivateCutOff:2.5,thresholdCutOff:.5,thresholdBeta:5,disableValueScaling:!0},Sr={maxTracks:18,maxAge:1e3,minSimilarity:.2,keypointTrackerParams:{keypointConfidenceThreshold:.3,keypointFalloff:[.026,.025,.025,.035,.035,.079,.079,.072,.072,.062,.062,.107,.107,.087,.087,.089,.089],minNumberOfKeypoints:4}},ha={maxTracks:18,maxAge:1e3,minSimilarity:.15,trackerParams:{}};function Of(s,e,a,o){for(var i={},t=0,u=gt;t<u.length;t++){var c=u[t];i[c]=[e[a[c]].y*o.height,e[a[c]].x*o.width]}if(function(k,I){return(k[I.left_hip].score>.2||k[I.right_hip].score>.2)&&(k[I.left_shoulder].score>.2||k[I.right_shoulder].score>.2)}(e,a)){var p=(i.left_hip[0]+i.right_hip[0])/2,h=(i.left_hip[1]+i.right_hip[1])/2,r=function(k,I,R,N,E){for(var O=["left_shoulder","right_shoulder","left_hip","right_hip"],j=0,B=0,D=0;D<O.length;D++)(Q=Math.abs(N-R[O[D]][0]))>j&&(j=Q),(ee=Math.abs(E-R[O[D]][1]))>B&&(B=ee);for(var V=0,q=0,$=0,Y=Object.keys(R);$<Y.length;$++){var Q,ee,ce=Y[$];k[I[ce]].score<.2||((Q=Math.abs(N-R[ce][0]))>V&&(V=Q),(ee=Math.abs(E-R[ce][1]))>q&&(q=ee))}return[j,B,V,q]}(e,a,i,p,h),f=r[0],b=r[1],g=r[2],_=r[3],v=Math.max(1.9*b,1.9*f,1.2*g,1.2*_),x=[p-(v=Math.min(v,Math.max(h,o.width-h,p,o.height-p))),h-v];if(v>Math.max(o.width,o.height)/2)return Lr(s==null,o);var S=2*v;return{yMin:x[0]/o.height,xMin:x[1]/o.width,yMax:(x[0]+S)/o.height,xMax:(x[1]+S)/o.width,height:(x[0]+S)/o.height-x[0]/o.height,width:(x[1]+S)/o.width-x[1]/o.width}}return Lr(s==null,o)}function Lr(s,e){var a,o,i,t;return s?e.width>e.height?(a=1,o=e.height/e.width,i=0,t=(e.width/2-e.height/2)/e.width):(a=e.width/e.height,o=1,i=(e.height/2-e.width/2)/e.height,t=0):e.width>e.height?(a=e.width/e.height,o=1,i=(e.height/2-e.width/2)/e.height,t=0):(a=1,o=e.height/e.width,i=0,t=(e.width/2-e.height/2)/e.width),{yMin:i,xMin:t,yMax:i+a,xMax:t+o,height:a,width:o}}function Pf(s){var e,a=s==null?Zi:de({},s);if(a.modelType==null)a.modelType="SinglePose.Lightning";else if(ca.indexOf(a.modelType)<0)throw new Error("Invalid architecture "+a.modelType+". Should be one of "+ca);if(a.enableSmoothing==null&&(a.enableSmoothing=!0),a.minPoseScore!=null&&(a.minPoseScore<0||a.minPoseScore>1))throw new Error("minPoseScore should be between 0.0 and 1.0");if(a.multiPoseMaxDimension!=null&&(a.multiPoseMaxDimension%32!=0||a.multiPoseMaxDimension<32))throw new Error("multiPoseMaxDimension must be a multiple of 32 and higher than 0");if(a.modelType==="MultiPose.Lightning"&&a.enableTracking==null&&(a.enableTracking=!0),a.modelType==="MultiPose.Lightning"&&a.enableTracking===!0)if(a.trackerType==null&&(a.trackerType=Xt.BoundingBox),a.trackerType===Xt.Keypoint)a.trackerConfig!=null?a.trackerConfig=function(o){var i=fa(Sr,o);return i.keypointTrackerParams=de({},Sr.keypointTrackerParams),o.keypointTrackerParams!=null&&(o.keypointTrackerParams.keypointConfidenceThreshold!=null&&(i.keypointTrackerParams.keypointConfidenceThreshold=o.keypointTrackerParams.keypointConfidenceThreshold),o.keypointTrackerParams.keypointFalloff!=null&&(i.keypointTrackerParams.keypointFalloff=o.keypointTrackerParams.keypointFalloff),o.keypointTrackerParams.minNumberOfKeypoints!=null&&(i.keypointTrackerParams.minNumberOfKeypoints=o.keypointTrackerParams.minNumberOfKeypoints)),i}(a.trackerConfig):a.trackerConfig=Sr;else{if(a.trackerType!==Xt.BoundingBox)throw new Error("Tracker type not supported by MoveNet");a.trackerConfig!=null?a.trackerConfig=(e=a.trackerConfig,fa(ha,e)):a.trackerConfig=ha}return a}function fa(s,e){var a={maxTracks:s.maxTracks,maxAge:s.maxAge,minSimilarity:s.minSimilarity};return e.maxTracks!=null&&(a.maxTracks=e.maxTracks),e.maxAge!=null&&(a.maxAge=e.maxAge),e.minSimilarity!=null&&(a.minSimilarity=e.minSimilarity),a}var Df=function(){function s(e,a){this.moveNetModel=e,this.modelInputResolution={height:0,width:0},this.keypointIndexByName=Nf(mt.MoveNet),a.modelType==="SinglePose.Lightning"?(this.modelInputResolution.width=192,this.modelInputResolution.height=192):a.modelType==="SinglePose.Thunder"&&(this.modelInputResolution.width=256,this.modelInputResolution.height=256),this.multiPoseModel=a.modelType==="MultiPose.Lightning",this.multiPoseModel||(this.keypointFilter=new Br(da),this.cropRegionFilterYMin=new Ft(.9),this.cropRegionFilterXMin=new Ft(.9),this.cropRegionFilterYMax=new Ft(.9),this.cropRegionFilterXMax=new Ft(.9)),this.enableSmoothing=a.enableSmoothing,a.minPoseScore?this.minPoseScore=a.minPoseScore:this.minPoseScore=.25,a.multiPoseMaxDimension?this.multiPoseMaxDimension=a.multiPoseMaxDimension:this.multiPoseMaxDimension=256,this.enableTracking=a.enableTracking,this.multiPoseModel&&this.enableTracking&&(a.trackerType===Xt.Keypoint?this.tracker=new Cf(a.trackerConfig):a.trackerType===Xt.BoundingBox&&(this.tracker=new Ef(a.trackerConfig)),this.enableSmoothing&&(this.keypointFilterMap=new Map))}return s.prototype.runSinglePersonPoseModel=function(e){return ie(this,void 0,void 0,function(){var a,o,i,t,u;return oe(this,function(c){switch(c.label){case 0:if((a=this.moveNetModel.execute(e)).shape.length!==4||a.shape[0]!==1||a.shape[1]!==1||a.shape[2]!==17||a.shape[3]!==3)throw a.dispose(),new Error("Unexpected output shape from model: ["+a.shape+"]");return Zn()==="webgpu"?[3,1]:(o=a.dataSync(),[3,3]);case 1:return[4,a.data()];case 2:o=c.sent(),c.label=3;case 3:for(a.dispose(),i={keypoints:[],score:0},t=0,u=0;u<17;++u)i.keypoints[u]={y:o[3*u],x:o[3*u+1],score:o[3*u+2]},i.keypoints[u].score>.2&&(++t,i.score+=i.keypoints[u].score);return t>0&&(i.score/=t),[2,i]}})})},s.prototype.runMultiPersonPoseModel=function(e){return ie(this,void 0,void 0,function(){var a,o,i,t,u,c,p,h;return oe(this,function(r){switch(r.label){case 0:if((a=this.moveNetModel.execute(e)).shape.length!==3||a.shape[0]!==1||a.shape[2]!==56)throw a.dispose(),new Error("Unexpected output shape from model: ["+a.shape+"]");return Zn()==="webgpu"?[3,1]:(o=a.dataSync(),[3,3]);case 1:return[4,a.data()];case 2:o=r.sent(),r.label=3;case 3:for(a.dispose(),i=[],t=o.length/56,u=0;u<t;++u)for(i[u]={keypoints:[]},c=56*u+51,i[u].box={yMin:o[c],xMin:o[c+1],yMax:o[c+2],xMax:o[c+3],width:o[c+3]-o[c+1],height:o[c+2]-o[c]},p=56*u+55,i[u].score=o[p],i[u].keypoints=[],h=0;h<17;++h)i[u].keypoints[h]={y:o[56*u+3*h],x:o[56*u+3*h+1],score:o[56*u+3*h+2]};return[2,i]}})})},s.prototype.estimatePoses=function(e,a,o){return a===void 0&&(a=pa),ie(this,void 0,void 0,function(){var i,t,u,c,p,h;return oe(this,function(r){switch(r.label){case 0:return a=function(f){return f==null?pa:de({},f)}(a),e==null?(this.reset(),[2,[]]):(o==null?Yi(e)&&(o=1e6*e.currentTime):o*=1e3,i=zr(e),t=Nn(i),u=Rn(i,0),e instanceof gn||i.dispose(),c=[],this.multiPoseModel?[3,2]:[4,this.estimateSinglePose(u,t,o)]);case 1:return c=r.sent(),[3,4];case 2:return[4,this.estimateMultiplePoses(u,t,o)];case 3:c=r.sent(),r.label=4;case 4:for(p=0;p<c.length;++p)for(h=0;h<c[p].keypoints.length;++h)c[p].keypoints[h].name=gt[h],c[p].keypoints[h].y*=t.height,c[p].keypoints[h].x*=t.width;return[2,c]}})})},s.prototype.estimateSinglePose=function(e,a,o){return ie(this,void 0,void 0,function(){var i,t,u,c,p=this;return oe(this,function(h){switch(h.label){case 0:return this.cropRegion||(this.cropRegion=Lr(this.cropRegion==null,a)),i=Pe(function(){var r=qt([[p.cropRegion.yMin,p.cropRegion.xMin,p.cropRegion.yMax,p.cropRegion.xMax]]),f=Gu([1],"int32"),b=[p.modelInputResolution.height,p.modelInputResolution.width];return Fn(Kt.cropAndResize(e,r,f,b,"bilinear",0),"int32")}),e.dispose(),[4,this.runSinglePersonPoseModel(i)];case 1:if(t=h.sent(),i.dispose(),t.score<this.minPoseScore)return this.reset(),[2,[]];for(u=0;u<t.keypoints.length;++u)t.keypoints[u].y=this.cropRegion.yMin+t.keypoints[u].y*this.cropRegion.height,t.keypoints[u].x=this.cropRegion.xMin+t.keypoints[u].x*this.cropRegion.width;return o!=null&&this.enableSmoothing&&(t.keypoints=this.keypointFilter.apply(t.keypoints,o,1)),c=Of(this.cropRegion,t.keypoints,this.keypointIndexByName,a),this.cropRegion=this.filterCropRegion(c),[2,[t]]}})})},s.prototype.estimateMultiplePoses=function(e,a,o){return ie(this,void 0,void 0,function(){var i,t,u,c,p,h,r,f,b,g,_,v=this;return oe(this,function(x){switch(x.label){case 0:return a.width>a.height?(t=this.multiPoseMaxDimension,u=Math.round(this.multiPoseMaxDimension*a.height/a.width),i=Kt.resizeBilinear(e,[u,t]),p=t,h=32*Math.ceil(u/32),c=Rr(i,[[0,0],[0,h-u],[0,0],[0,0]])):(t=Math.round(this.multiPoseMaxDimension*a.width/a.height),u=this.multiPoseMaxDimension,i=Kt.resizeBilinear(e,[u,t]),p=32*Math.ceil(t/32),h=u,c=Rr(i,[[0,0],[0,0],[0,p-t],[0,0]])),i.dispose(),e.dispose(),r=Fn(c,"int32"),c.dispose(),[4,this.runMultiPersonPoseModel(r)];case 1:for(f=x.sent(),r.dispose(),f=f.filter(function(S){return S.score>=v.minPoseScore}),g=0;g<f.length;++g)for(b=0;b<f[g].keypoints.length;++b)f[g].keypoints[b].y*=h/u,f[g].keypoints[b].x*=p/t;if(this.enableTracking&&(this.tracker.apply(f,o),this.enableSmoothing)){for(g=0;g<f.length;++g)this.keypointFilterMap.has(f[g].id)||this.keypointFilterMap.set(f[g].id,new Br(da)),f[g].keypoints=this.keypointFilterMap.get(f[g].id).apply(f[g].keypoints,o,1);_=this.tracker.getTrackIDs(),this.keypointFilterMap.forEach(function(S,k){_.has(k)||v.keypointFilterMap.delete(k)})}return[2,f]}})})},s.prototype.filterCropRegion=function(e){if(e){var a=this.cropRegionFilterYMin.apply(e.yMin),o=this.cropRegionFilterXMin.apply(e.xMin),i=this.cropRegionFilterYMax.apply(e.yMax),t=this.cropRegionFilterXMax.apply(e.xMax);return{yMin:a,xMin:o,yMax:i,xMax:t,height:i-a,width:t-o}}return this.cropRegionFilterYMin.reset(),this.cropRegionFilterXMin.reset(),this.cropRegionFilterYMax.reset(),this.cropRegionFilterXMax.reset(),null},s.prototype.dispose=function(){this.moveNetModel.dispose()},s.prototype.reset=function(){this.cropRegion=null,this.resetFilters()},s.prototype.resetFilters=function(){this.keypointFilter.reset(),this.cropRegionFilterYMin.reset(),this.cropRegionFilterXMin.reset(),this.cropRegionFilterYMax.reset(),this.cropRegionFilterXMax.reset()},s}();function Bf(s){return s===void 0&&(s=Zi),ie(this,void 0,void 0,function(){var e,a,o,i;return oe(this,function(t){switch(t.label){case 0:return e=Pf(s),o=!0,e.modelUrl?(o=typeof e.modelUrl=="string"&&e.modelUrl.indexOf("https://tfhub.dev")>-1,[4,fn(e.modelUrl,{fromTFHub:o})]):[3,2];case 1:return a=t.sent(),[3,4];case 2:return i=void 0,e.modelType==="SinglePose.Lightning"?i="https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4":e.modelType==="SinglePose.Thunder"?i="https://tfhub.dev/google/tfjs-model/movenet/singlepose/thunder/4":e.modelType==="MultiPose.Lightning"&&(i="https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1"),[4,fn(i,{fromTFHub:o})];case 3:a=t.sent(),t.label=4;case 4:return Zn()==="webgl"&&Jn().set("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD",0),[2,new Df(a,e)]}})})}var ma={architecture:"MobileNetV1",outputStride:16,multiplier:.75,inputResolution:{height:257,width:257}},ga=["MobileNetV1","ResNet50"],ya={MobileNetV1:[8,16],ResNet50:[16]},Lf=[8,16,32],ba={MobileNetV1:[.5,.75,1],ResNet50:[1]},_a=[1,2,4],jf={maxPoses:1,flipHorizontal:!1},Wf={maxPoses:5,flipHorizontal:!1,scoreThreshold:.5,nmsRadius:20},Hf=[-123.15,-115.9,-103.06];function Mr(s){return Math.floor(s/2)}var Vf=function(){function s(e,a){this.priorityQueue=new Array(e),this.numberOfElements=-1,this.getElementValue=a}return s.prototype.enqueue=function(e){this.priorityQueue[++this.numberOfElements]=e,this.swim(this.numberOfElements)},s.prototype.dequeue=function(){var e=this.priorityQueue[0];return this.exchange(0,this.numberOfElements--),this.sink(0),this.priorityQueue[this.numberOfElements+1]=null,e},s.prototype.empty=function(){return this.numberOfElements===-1},s.prototype.size=function(){return this.numberOfElements+1},s.prototype.all=function(){return this.priorityQueue.slice(0,this.numberOfElements+1)},s.prototype.max=function(){return this.priorityQueue[0]},s.prototype.swim=function(e){for(;e>0&&this.less(Mr(e),e);)this.exchange(e,Mr(e)),e=Mr(e)},s.prototype.sink=function(e){for(;2*e<=this.numberOfElements;){var a=2*e;if(a<this.numberOfElements&&this.less(a,a+1)&&a++,!this.less(e,a))break;this.exchange(e,a),e=a}},s.prototype.getValueAt=function(e){return this.getElementValue(this.priorityQueue[e])},s.prototype.less=function(e,a){return this.getValueAt(e)<this.getValueAt(a)},s.prototype.exchange=function(e,a){var o=this.priorityQueue[e];this.priorityQueue[e]=this.priorityQueue[a],this.priorityQueue[a]=o},s}();function Uf(s,e,a,o,i,t){for(var u=t.shape,c=u[0],p=u[1],h=!0,r=Math.max(a-i,0),f=Math.min(a+i+1,c),b=r;b<f;++b){for(var g=Math.max(o-i,0),_=Math.min(o+i+1,p),v=g;v<_;++v)if(t.get(b,v,s)>e){h=!1;break}if(!h)break}return h}function zf(s){return ie(this,void 0,void 0,function(){return oe(this,function(e){return[2,Promise.all(s.map(function(a){return a.buffer()}))]})})}function eo(s,e,a,o){return{y:o.get(s,e,a),x:o.get(s,e,a+17)}}function to(s,e,a){var o=eo(s.heatmapY,s.heatmapX,s.id,a),i=o.y,t=o.x;return{x:s.heatmapX*e+t,y:s.heatmapY*e+i}}function no(s,e,a,o){var i=a.x,t=a.y;return s.some(function(u){var c,p,h,r,f,b,g=u.keypoints;return c=t,p=i,h=g[o].y,r=g[o].x,(f=h-c)*f+(b=r-p)*b<=e})}var va=gt.reduce(function(s,e,a){return s[e]=a,s},{}),ro=[["nose","left_eye"],["left_eye","left_ear"],["nose","right_eye"],["right_eye","right_ear"],["nose","left_shoulder"],["left_shoulder","left_elbow"],["left_elbow","left_wrist"],["left_shoulder","left_hip"],["left_hip","left_knee"],["left_knee","left_ankle"],["nose","right_shoulder"],["right_shoulder","right_elbow"],["right_elbow","right_wrist"],["right_shoulder","right_hip"],["right_hip","right_knee"],["right_knee","right_ankle"]].map(function(s){var e=s[0],a=s[1];return[va[e],va[a]]}),Ar=ro.map(function(s){return s[1]}),wa=ro.map(function(s){return s[0]});function ka(s,e,a){return s<e?e:s>a?a:s}function Tr(s,e,a,o){return{y:ka(Math.round(s.y/e),0,a-1),x:ka(Math.round(s.x/e),0,o-1)}}function Ia(s,e){return{x:s.x+e.x,y:s.y+e.y}}function xa(s,e,a,o,i,t,u,c){c===void 0&&(c=2);for(var p=o.shape,h=p[0],r=p[1],f={y:e.y,x:e.x},b=Ia(f,function(k,I,R){var N=R.shape[2]/2;return{y:R.get(I.y,I.x,k),x:R.get(I.y,I.x,N+k)}}(s,Tr(f,t,h,r),u)),g=0;g<c;g++){var _=Tr(b,t,h,r),v=eo(_.y,_.x,a,i);b=Ia({x:_.x*t,y:_.y*t},{x:v.x,y:v.y})}var x=Tr(b,t,h,r),S=o.get(x.y,x.x,a);return{y:b.y,x:b.x,name:gt[a],score:S}}function $f(s,e,a,o,i,t){var u=e.shape[2],c=Ar.length,p=new Array(u),h=s.part,r=s.score,f=to(h,o,a);p[h.id]={score:r,name:gt[h.id],y:f.y,x:f.x};for(var b=c-1;b>=0;--b){var g=Ar[b],_=wa[b];p[g]&&!p[_]&&(p[_]=xa(b,p[g],_,e,a,o,t))}for(b=0;b<c;++b)g=wa[b],_=Ar[b],p[g]&&!p[_]&&(p[_]=xa(b,p[g],_,e,a,o,i));return p}function Gf(s,e,a){return a.reduce(function(o,i,t){var u=i.y,c=i.x,p=i.score;return no(s,e,{y:u,x:c},t)||(o+=p),o},0)/a.length}function qf(s,e,a,o,i,t,u,c){return u===void 0&&(u=.5),c===void 0&&(c=20),ie(this,void 0,void 0,function(){var p,h,r,f,b,g,_,v,x,S,k,I;return oe(this,function(R){switch(R.label){case 0:return[4,zf([s,e,a,o])];case 1:for(p=R.sent(),h=p[0],r=p[1],f=p[2],b=p[3],g=[],_=function(N,E,O){for(var j=O.shape,B=j[0],D=j[1],V=j[2],q=new Vf(B*D*V,function(ce){return ce.score}),$=0;$<B;++$)for(var Y=0;Y<D;++Y)for(var Q=0;Q<V;++Q){var ee=O.get($,Y,Q);ee<N||Uf(Q,ee,$,Y,E,O)&&q.enqueue({score:ee,part:{heatmapY:$,heatmapX:Y,id:Q}})}return q}(u,1,h),v=c*c;g.length<t&&!_.empty();)x=_.dequeue(),S=to(x.part,i,r),no(g,v,S,x.part.id)||(k=$f(x,h,r,i,f,b),I=Gf(g,v,k),g.push({keypoints:k,score:I}));return[2,g]}})})}function Kf(s){var e=s.shape,a=e[0],o=e[1],i=e[2];return Pe(function(){var t,u,c=Rt(s,[a*o,i]),p=Qu(c,0),h=Rn(Ue(p,Qn(o,"int32")),1),r=Rn((t=p,u=o,Pe(function(){var f=Ue(t,Qn(u,"int32"));return hn(t,Me(f,Qn(u,"int32")))})),1);return Fr([h,r],1)})}function Xf(s,e,a){return Pe(function(){var o=function(i,t){for(var u=[],c=0;c<gt.length;c++){var p=i.get(c,0).valueOf(),h=i.get(c,1).valueOf(),r=Yf(p,h,c,t),f=r.x,b=r.y;u.push(b),u.push(f)}return qt(u,[gt.length,2])}(s,a);return ze(Fn(Me(s.toTensor(),Qn(e,"int32")),"float32"),o)})}function Yf(s,e,a,o){return{y:o.get(s,e,a),x:o.get(s,e,a+gt.length)}}function Qf(s,e,a){return ie(this,void 0,void 0,function(){var o,i,t,u,c,p,h,r,f,b;return oe(this,function(g){switch(g.label){case 0:return o=0,i=Kf(s),[4,Promise.all([s.buffer(),e.buffer(),i.buffer()])];case 1:return t=g.sent(),u=t[0],c=t[1],p=t[2],[4,(h=Xf(p,a,c)).buffer()];case 2:return r=g.sent(),f=Array.from(function(_,v){for(var x=v.shape[0],S=new Float32Array(x),k=0;k<x;k++){var I=v.get(k,0),R=v.get(k,1);S[k]=_.get(I,R,k)}return S}(u,p)),b=f.map(function(_,v){return o+=_,{y:r.get(v,0),x:r.get(v,1),score:_,name:gt[v]}}),i.dispose(),h.dispose(),[2,{keypoints:b,score:o/b.length}]}})})}function Sa(s,e){return(s-1)%e==0}var Ma="https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/",Aa="https://storage.googleapis.com/tfjs-models/savedmodel/posenet/resnet50/";function Ta(s,e){return function(a,o){return(a-1)%o==0}(s,e)?s:Math.floor(s/e)*e+1}var Ra=function(){function s(e,a){this.posenetModel=e;var o=this.posenetModel.inputs[0].shape;Be(o[1]===-1&&o[2]===-1,function(){return"Input shape ["+o[1]+", "+o[2]+"] must both be equal to or -1"});var i,t,u=(i=a.inputResolution,t=a.outputStride,{height:Ta(i.height,t),width:Ta(i.width,t)});(function(c){Be(Lf.indexOf(c)>=0,function(){return"outputStride of "+c+" is invalid. It must be either 8 or 16."})})(a.outputStride),function(c,p){Be(Sa(c.height,p),function(){return"height of "+c.height+" is invalid for output stride "+p+"."}),Be(Sa(c.width,p),function(){return"width of "+c.width+" is invalid for output stride "+p+"."})}(u,a.outputStride),this.inputResolution=u,this.outputStride=a.outputStride,this.architecture=a.architecture}return s.prototype.estimatePoses=function(e,a){return a===void 0&&(a=jf),ie(this,void 0,void 0,function(){var o,i,t,u,c,p,h,r,f,b,g,_,v,x,S;return oe(this,function(k){switch(k.label){case 0:return o=function(I){var R=I;if(R.maxPoses==null&&(R.maxPoses=1),R.maxPoses<=0)throw new Error("Invalid maxPoses "+R.maxPoses+". Should be > 0.");if(R.maxPoses>1){if((R=de(de({},Wf),R)).scoreThreshold<0||R.scoreThreshold>1)throw new Error("Invalid scoreThreshold "+R.scoreThreshold+". Should be in range [0.0, 1.0]");if(R.nmsRadius<=0)throw new Error("Invalid nmsRadius "+R.nmsRadius+".")}return R}(a),e==null?[2,[]]:(this.maxPoses=o.maxPoses,i=Dr(e,{outputTensorSize:this.inputResolution,keepAspectRatio:!0,borderMode:"replicate"}),t=i.imageTensor,u=i.padding,c=this.architecture==="ResNet50"?ze(t,Hf):Xi(t,[-1,1]),p=this.posenetModel.predict(c),this.architecture==="ResNet50"?(h=me(p[2],[0]),r=me(p[3],[0]),f=me(p[0],[0]),b=me(p[1],[0])):(h=me(p[0],[0]),r=me(p[1],[0]),f=me(p[2],[0]),b=me(p[3],[0])),g=Vr(r),this.maxPoses!==1?[3,2]:[4,Qf(g,h,this.outputStride)]);case 1:return v=k.sent(),_=[v],[3,4];case 2:return[4,qf(g,h,f,b,this.outputStride,this.maxPoses,o.scoreThreshold,o.nmsRadius)];case 3:_=k.sent(),k.label=4;case 4:return x=Nn(e),S=function(I,R,N,E){var O=R.height,j=R.width,B=O/(N.height*(1-E.top-E.bottom)),D=j/(N.width*(1-E.left-E.right)),V=-E.top*N.height,q=-E.left*N.width;if(D===1&&B===1&&V===0&&q===0)return I;for(var $=0,Y=I;$<Y.length;$++)for(var Q=0,ee=Y[$].keypoints;Q<ee.length;Q++){var ce=ee[Q];ce.x=(ce.x+q)*D,ce.y=(ce.y+V)*B}return I}(_,x,this.inputResolution,u),o.flipHorizontal&&(S=function(I,R){for(var N=0,E=I;N<E.length;N++)for(var O=0,j=E[N].keypoints;O<j.length;O++){var B=j[O];B.x=R.width-1-B.x}return I}(S,x)),t.dispose(),c.dispose(),De(p),h.dispose(),r.dispose(),f.dispose(),b.dispose(),g.dispose(),[2,S]}})})},s.prototype.dispose=function(){this.posenetModel.dispose()},s.prototype.reset=function(){},s}();function Jf(s){return s===void 0&&(s=ma),ie(this,void 0,void 0,function(){var e,a,o,i,t;return oe(this,function(u){switch(u.label){case 0:return(e=function(r){var f=r||ma;if(f.architecture==null&&(f.architecture="MobileNetV1"),ga.indexOf(f.architecture)<0)throw new Error("Invalid architecture "+f.architecture+". Should be one of "+ga);if(f.inputResolution==null&&(f.inputResolution={height:257,width:257}),f.outputStride==null&&(f.outputStride=16),ya[f.architecture].indexOf(f.outputStride)<0)throw new Error("Invalid outputStride "+f.outputStride+". Should be one of "+ya[f.architecture]+" for architecture "+f.architecture+".");if(f.multiplier==null&&(f.multiplier=1),ba[f.architecture].indexOf(f.multiplier)<0)throw new Error("Invalid multiplier "+f.multiplier+". Should be one of "+ba[f.architecture]+" for architecture "+f.architecture+".");if(f.quantBytes==null&&(f.quantBytes=4),_a.indexOf(f.quantBytes)<0)throw new Error("Invalid quantBytes "+f.quantBytes+". Should be one of "+_a+" for architecture "+f.architecture+".");if(f.architecture==="MobileNetV1"&&f.outputStride===32&&f.multiplier!==1)throw new Error("When using an output stride of 32, you must select 1 as the multiplier.");return f}(s)).architecture!=="ResNet50"?[3,2]:(c=e.outputStride,p=e.quantBytes,h="model-stride"+c+".json",a=p===4?Aa+"float/"+h:Aa+"quant"+p+"/"+h,[4,fn(e.modelUrl||a)]);case 1:return o=u.sent(),[2,new Ra(o,e)];case 2:return i=function(r,f,b){var g={1:"100",.75:"075",.5:"050"},_="model-stride"+r+".json";return b===4?Ma+"float/"+g[f]+"/"+_:Ma+"quant"+b+"/"+g[f]+"/"+_}(e.outputStride,e.multiplier,e.quantBytes),[4,fn(e.modelUrl||i)];case 3:return t=u.sent(),[2,new Ra(t,e)]}var c,p,h})})}function Zf(s,e){return ie(this,void 0,void 0,function(){var a,o;return oe(this,function(i){switch(s){case mt.PoseNet:return[2,Jf(e)];case mt.BlazePose:if(o=void 0,(a=e)!=null){if(a.runtime==="tfjs")return[2,Ff(e)];if(a.runtime==="mediapipe")return[2,tf(e)];o=a.runtime}throw new Error("Expect modelConfig.runtime to be either 'tfjs' or 'mediapipe', but got "+o);case mt.MoveNet:return[2,Bf(e)];default:throw new Error(s+" is not a supported model name.")}})})}const ao=s=>(il("data-v-e1066a71"),s=s(),ol(),s),em={class:"FaceLandmarksDetection"},tm={class:"box"},nm=ao(()=>$t("h3",null,"输入",-1)),rm=ao(()=>$t("h3",null,"输出",-1)),am=nl({__name:"PoseDetection",setup(s){const e=[["left_ear","left_eye","nose","right_eye","right_ear"],["left_wrist","left_elbow","left_shoulder","right_shoulder","right_elbow","right_wrist"],["left_shoulder","left_hip","left_knee","left_ankle"],["right_shoulder","right_hip","right_knee","right_ankle"],["left_hip","right_hip"]],a=Yn(!1),o=Yn(.5);let i;const t=Yn(),{videoConfig:u,videoButtonClick:c,newVideoRef:p}=el({videoProceed:_,videoRef:t}),h=Yn(),{getImageData:r,drawImage:f,fillArc:b,drawLine:g}=tl({canvasRef:h});async function _(){if(!u.status)return;f(p.value);const x=r();(await i.estimatePoses(x,{scoreThreshold:o.value})).forEach(k=>{const I={};k.keypoints.forEach(R=>{R.score>o.value&&b(R.x,R.y,1,"#409eff"),I[R.name]=R}),e.forEach(R=>{const N=R.map(E=>{const{x:O,y:j,score:B}=I[E];return B>o.value?[O,j]:null}).filter(E=>E);N.length>0&&g({data:N,color:"#67c23a"})})}),window.requestAnimationFrame(_)}async function v(){const x=mt.MoveNet;i=await Zf(x),a.value=!0}return rl(async()=>{Ju("wasm").then(()=>v())}),(x,S)=>(Jr(),Qr("div",em,[$t("div",tm,[$t("div",null,[nm,$t("video",{id:"video",ref_key:"videoRef",ref:t,autoplay:""},null,512)]),$t("div",null,[rm,$t("canvas",{id:"canvasRef",ref_key:"canvasRef",ref:h},null,512)])]),a.value?(Jr(),Qr("button",{key:0,onClick:S[0]||(S[0]=(...k)=>yr(c)&&yr(c)(...k))},al(yr(u).status?"关闭":"开启")+"摄像头 ",1)):sl("",!0)]))}});const cm=Zu(am,[["__scopeId","data-v-e1066a71"]]);export{cm as default};
