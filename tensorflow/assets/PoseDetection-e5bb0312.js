import{aP as Et,aN as An,aQ as Ui}from"./index-c5cb974d.js";import"./register_all_kernels-49f39960.js";import{_ as ga,w as xr,x as ee,A as zi,y as $i,z as ya,I as Gi,B as ba,C as Cn,D as Mr,E as yn,F as _a,G as Nt,H as un,J as ln,K as va,L as wa,M as ka,N as Ia,R as qi,O as Ki,g as et,P as Sa,S as Xi,Q as ht,U as Yi,V as Qi,W as Ji,X as xa,Y as Zi,Z as Ma,$ as Aa,a0 as On,a1 as Ta,a2 as eo,a3 as to,a4 as Ra,a5 as Fa,a6 as Ea,a7 as Na,a8 as Ca,a9 as Oa,aa as no,ab as ro,ac as ao,ad as so,ae as io,af as Pa,ag as oo,ah as uo,ai as Da,aj as La,ak as Ba,al as ja,am as lo,an as co,ao as po,ap as Va,aq as ho,ar as fo,as as mo,at as go,au as yo,av as bo,aw as _o,ax as Wa,ay as vo,az as Ha,aA as Ua,aB as za,aC as wo,aD as $a,aE as ko,aF as Io,aG as Ga,aH as qa,aI as Ka,aJ as So,aK as Xa,aL as xo,aM as Mo,aN as Ao,aO as Ya,aP as To,aQ as Qa,aR as Ja,aS as Ro,aT as Fo,aU as Eo,aV as No,aW as Za,aX as es,aY as ts,aZ as ns,a_ as Co,a$ as Oo,b0 as rs,b1 as Po,b2 as Do,b3 as Lo,b4 as Bo,b5 as as,b6 as jo,b7 as Vo,b8 as ss,b9 as Wo,ba as Ho,bb as Uo,bc as zo,bd as $o,be as Go,bf as is,bg as os,bh as qo,bi as Ko,bj as Xo,bk as Yo,bl as us,bm as Qo,bn as Jo,bo as ls,bp as cs,bq as ps,br as Zo,bs as eu,bt as tu,bu as Hn,bv as nu,bw as ru,bx as ds,by as Cr,bz as Or,bA as au,l as gn,bB as Un,T as bn,t as Ar,j as Mn,b as rn,a as Ye,e as Tn,bC as mr,bD as su,d as Ze,c as Rn,s as Re,i as an,r as Vt,v as Tr,bE as iu,h as ot,f as hs,m as $e,k as st,bF as ou,p as mn,bG as uu,bH as lu,n as it,o as Pr,q as gr,u as cu,bI as pu,bJ as Wn}from"./graph_model-9162a669.js";import{p as du,i as hu,c as fu,s as Dr,g as fs,a as ms,b as gs,d as mu,e as gu,f as yu,h as nr,j as bu,k as _u,l as vu,m as wu,r as ku,n as Iu,o as Su,q as xu,t as Mu,u as Au,v as Tu,w as Ru,x as Fu,y as Eu,z as Lr,A as Nu,B as Cu,C as Ou,D as Pu,E as Du,F as Lu,G as Bu,H as ju,I as Vu,J as Wu}from"./useCanvas-c9b0a34e.js";import{u as Hu}from"./useVideo-bd9c37bb.js";import{d as Uu,f as Vn,h as zu,c as Br,i as tn,u as rr,t as $u,j as Gu,o as jr,p as qu,l as Ku}from"./index-7b9c9dd6.js";import{_ as Xu}from"./_plugin-vue_export-helper-c27b6911.js";function ys(a,e){for(var r=0;r<e.length;r++){const i=e[r];if(typeof i!="string"&&!Array.isArray(i)){for(const s in i)if(s!=="default"&&!(s in a)){const t=Object.getOwnPropertyDescriptor(i,s);t&&Object.defineProperty(a,s,t.get?t:{enumerable:!0,get:()=>i[s]})}}}return Object.freeze(Object.defineProperty(a,Symbol.toStringTag,{value:"Module"}))}/**
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
 */var Ee;(function(a){a[a.float32=0]="float32",a[a.int32=1]="int32",a[a.bool=2]="bool",a[a.string=3]="string",a[a.complex64=4]="complex64"})(Ee||(Ee={}));var Fn;(function(a){a[a.linear=0]="linear",a[a.relu=1]="relu",a[a.relu6=2]="relu6",a[a.prelu=3]="prelu",a[a.leakyrelu=4]="leakyrelu",a[a.sigmoid=5]="sigmoid",a[a.elu=6]="elu"})(Fn||(Fn={}));/**
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
 */let bs;function Yu(a){bs=a.wasm.cwrap(ga,null,["number","array","number","number","array","number","number","number","number","number","number","number","number"])}function Qu(a){const{inputs:e,backend:r,attrs:i}=a,{a:s,b:t,bias:u,preluActivationWeights:p}=e;if(s.dtype!=="float32"||t.dtype!=="float32")throw new Error("_FusedMatMul for non non-float32 tensors not yet supported.");const{transposeA:h,transposeB:d,activation:n,leakyreluAlpha:f}=i,_=r.dataIdMap.get(s.dataId).id,m=r.dataIdMap.get(t.dataId).id;let y=0;if(u!=null){const V=r.dataIdMap.get(u.dataId);if(V.shape.length!==1)throw new Error(`_FusedMatMul only supports rank-1 bias but got rank ${V.shape.length}.`);y=V.id}const b=p==null?0:r.dataIdMap.get(p.dataId).id,k=Fn[n];if(k==null)throw new Error(`${n} activation not yet supported for FusedConv2D in the wasm backend.`);const S=h?s.shape[2]:s.shape[1],I=d?t.shape[1]:t.shape[2],w=xr(s.shape.slice(0,-2),t.shape.slice(0,-2)),A=r.makeOutput([...w,S,I],s.dtype),O=r.dataIdMap.get(A.dataId).id,F=new Uint8Array(new Int32Array(s.shape).buffer),P=new Uint8Array(new Int32Array(t.shape).buffer);return bs(_,F,s.shape.length,m,P,t.shape.length,h,d,k,y,b,f||0,O),A}const Ju={kernelName:ga,backendName:"wasm",setupFunc:Yu,kernelFunc:Qu};/**
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
 */function Ve(a,e){let r;function i(t){r=t.wasm.cwrap(a,null,["number","number","number"])}function s(t){const{backend:u,inputs:{x:p}}=t,h=u.dataIdMap.get(p.dataId).id,d=u.makeOutput(p.shape,e||p.dtype),n=u.dataIdMap.get(d.dataId).id;return ee(d.shape)===0||r(h,Ee[p.dtype],n),d}return{kernelName:a,backendName:"wasm",setupFunc:i,kernelFunc:s}}/**
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
 */const Zu=Ve(zi);/**
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
 */function Ge(a,e,r){let i;function s(u){i=u.wasm.cwrap(a,null,["number","array","number","number","array","number","number","number"])}function t(u){const{backend:p,inputs:h}=u,{a:d,b:n}=h,f=p.dataIdMap.get(d.dataId).id,_=p.dataIdMap.get(n.dataId).id,m=r??d.dtype,y=xr(d.shape,n.shape),b=p.makeOutput(y,m);if(ee(y)===0)return b;const k=new Uint8Array(new Int32Array(d.shape).buffer),S=new Uint8Array(new Int32Array(n.shape).buffer),I=p.dataIdMap.get(b.dataId).id;return(()=>i(f,k,d.shape.length,_,S,n.shape.length,Ee[d.dtype],I))(),b}return{kernelName:a,backendName:"wasm",setupFunc:s,kernelFunc:t}}/**
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
 */const el=Ge($i);/**
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
 */let _s;function tl(a){_s=a.wasm.cwrap(ya,null,["array","number","number","number"])}function nl(a){const{inputs:e,backend:r}=a,i=r.makeOutput(e[0].shape,e[0].dtype);if(ee(i.shape)===0)return i;const s=e.map(p=>r.dataIdMap.get(p.dataId).id),t=new Uint8Array(new Int32Array(s).buffer),u=r.dataIdMap.get(i.dataId).id;return _s(t,s.length,Ee[i.dtype],u),i}const rl={kernelName:ya,backendName:"wasm",setupFunc:tl,kernelFunc:nl};/**
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
 */function qn(a){const{inputs:{x:e},backend:r}=a,i=r.makeOutput(e.shape,e.dtype),s=r.typedArrayFromHeap(e);return r.typedArrayFromHeap(i).set(s),i}const al={kernelName:Gi,backendName:"wasm",kernelFunc:qn};/**
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
 */let vs;function sl(a){vs=a.wasm.cwrap(ba,null,["number","array","number","number","number","array","number"])}function Ht(a){const{inputs:e,backend:r,attrs:i}=a,[s,t]=ol(e.x.shape,i.perm);let u=!0;for(let y=0;y<t.length;y++)t[y]!==y&&(u=!1);const p=il(e.x.shape,i.perm),h={dataId:e.x.dataId,shape:s,dtype:e.x.dtype};if(u){const y=qn({inputs:e,backend:r});return y.shape=p,y}const d=r.makeOutput(p,h.dtype),n=r.dataIdMap.get(h.dataId).id,f=r.dataIdMap.get(d.dataId).id,_=new Uint8Array(new Int32Array(t).buffer),m=new Uint8Array(new Int32Array(h.shape).buffer);return vs(n,m,h.shape.length,Ee[h.dtype],f,_,t.length),d}function il(a,e){const r=new Array(a.length);for(let i=0;i<r.length;i++)r[i]=a[e[i]];return r}function ol(a,e){const r=[],i=[];for(let s=0;s<a.length;++s)a[s]!==1&&r.push(a[s]),a[e[s]]!==1&&i.push(e[s]);for(let s=0;s<i.length;++s){let t=-1;for(let u=0;u<i.length;++u)i[u]>=s&&(t===-1||i[t]>i[u])&&(t=u);i[t]=s}return[r,i]}const ul={kernelName:ba,backendName:"wasm",kernelFunc:Ht,setupFunc:sl};/**
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
 */function Ut(a,e,r){const i=a.shape,s=a.shape.length,t=Cn(e,i);let u=t;const p=Mr(u,s);let h=null,d=!1;if(p!=null){const n=new Array(s);for(let m=0;m<n.length;m++)n[m]=i[p[m]];u=yn(u.length,s),h=Ht({inputs:{x:a},attrs:{perm:p},backend:r});const f=r.dataIdMap.get(a.dataId).id;r.dataIdMap.get(h.dataId).id!==f&&(d=!0)}return{transposed:h,originalAxes:t,axes:u,inputWasTransposed:d}}/**
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
 */let ws;function ll(a){ws=a.wasm.cwrap(_a,null,["number, number, number"])}function cl(a){const{backend:e,inputs:r,attrs:i}=a,{axis:s,keepDims:t}=i,{x:u}=r;let h=e.dataIdMap.get(u.dataId).id,d=u;const{transposed:n,axes:f,originalAxes:_,inputWasTransposed:m}=Ut(u,s,e);if(m){const w=e.dataIdMap.get(n.dataId).id;d=n,h=w}const y=d.shape.length;Nt("all",f,y);const[b,k]=un(d.shape,f),S=ee(k),I=e.makeOutput(b,u.dtype);if(ee(d.shape)!==0){const w=e.dataIdMap.get(I.dataId).id;ws(h,S,w)}if(m&&e.disposeData(n.dataId),t){const w=ln(I.shape,_);I.shape=w}return I}const pl={kernelName:_a,backendName:"wasm",setupFunc:ll,kernelFunc:cl};/**
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
 */let ks;function dl(a){ks=a.wasm.cwrap(va,null,["number, number, number"])}function hl(a){const{backend:e,inputs:r,attrs:i}=a,{axis:s,keepDims:t}=i,{x:u}=r;let h=e.dataIdMap.get(u.dataId).id,d=u;const{transposed:n,axes:f,originalAxes:_,inputWasTransposed:m}=Ut(u,s,e);if(m){const w=e.dataIdMap.get(n.dataId).id;d=n,h=w}const y=d.shape.length;Nt("any",f,y);const[b,k]=un(d.shape,f),S=ee(k),I=e.makeOutput(b,u.dtype);if(ee(d.shape)!==0){const w=e.dataIdMap.get(I.dataId).id;ks(h,S,w)}if(m&&e.disposeData(n.dataId),t){const w=ln(I.shape,_);I.shape=w}return I}const fl={kernelName:va,backendName:"wasm",setupFunc:dl,kernelFunc:hl};/**
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
 */let Is;function ml(a){Is=a.wasm.cwrap(wa,null,["number","number","number","number","number"])}function gl(a){const{backend:e,inputs:r,attrs:i}=a,{axis:s}=i,{x:t}=r,u=e.dataIdMap.get(t.dataId).id;let p=u,h=t;const{transposed:d,axes:n,inputWasTransposed:f}=Ut(t,s,e);if(f){const S=e.dataIdMap.get(d.dataId).id;S!==u&&(h=d,p=S)}const _=h.shape.slice(0,-1),m=e.makeOutput(_,"int32"),y=e.dataIdMap.get(m.dataId).id,b=ee(m.shape),k=h.shape[n[0]];return Is(p,Ee[h.dtype],b,k,y),f&&e.disposeData(d.dataId),m}const yl={kernelName:wa,backendName:"wasm",kernelFunc:gl,setupFunc:ml};/**
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
 */let Ss;function bl(a){Ss=a.wasm.cwrap(ka,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function _l(a){const{inputs:e,attrs:r,backend:i}=a,s=e.x,t=i.dataIdMap.get(s.dataId).id,{filterSize:u,strides:p,pad:h,dimRoundingMode:d}=r,n=Ia(s.shape,u,p,1,h,d),f=n.filterHeight,_=n.filterWidth,m=n.padInfo.top,y=n.padInfo.right,b=n.padInfo.bottom,k=n.padInfo.left,S=n.strideHeight,I=n.strideWidth,w=n.inChannels;if(n.dataFormat!=="channelsLast")throw new Error(`wasm backend does not support dataFormat:'${n.dataFormat}'. Please use 'channelsLast'.`);if(n.dilationWidth!==1||n.dilationHeight!==1)throw new Error(`was backend only supports average pooling with dilation = [1, 1], got [${n.dilationHeight}, ${n.dilationWidth}].`);const A=i.makeOutput(n.outShape,"float32"),O=i.dataIdMap.get(A.dataId).id;return Ss(t,s.shape[0],s.shape[1],s.shape[2],f,_,m,y,b,k,S,I,w,O),A}const vl={kernelName:ka,backendName:"wasm",setupFunc:bl,kernelFunc:_l};/**
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
 */function tt(a){const{inputs:e,attrs:r}=a,{x:i}=e,{shape:s}=r,t=ee(i.shape),u=Ki(s,t);return et(t===ee(u),()=>`new shape: ${u}, old shape: ${i.shape}. New shape and old shape must have the same number of elements.`),a.backend.incRef(i.dataId),{dataId:i.dataId,shape:u,dtype:i.dtype}}const wl={kernelName:qi,backendName:"wasm",kernelFunc:tt};/**
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
 */let xs;function kl(a){xs=a.wasm.cwrap(Sa,null,["number","array","number","number","array","number","number","number","number"])}function Il(a){const{inputs:e,backend:r,attrs:i}=a,{a:s,b:t}=e,{transposeA:u,transposeB:p}=i;if(s.dtype!=="float32"||t.dtype!=="float32")throw new Error("BatchMatMul for non non-float32 tensors not yet supported.");const h=s.shape.length,d=t.shape.length,n=u?s.shape[h-2]:s.shape[h-1],f=p?t.shape[d-1]:t.shape[d-2],_=u?s.shape[h-1]:s.shape[h-2],m=p?t.shape[d-2]:t.shape[d-1],y=s.shape.slice(0,-2),b=t.shape.slice(0,-2),k=ee(y),S=ee(b),w=xr(s.shape.slice(0,-2),t.shape.slice(0,-2)).concat([_,m]);et(n===f,()=>`Error in matMul: inner shapes (${n}) and (${f}) of Tensors with shapes ${s.shape} and ${t.shape} and transposeA=${u} and transposeB=${p} must match.`);const A=u?[k,n,_]:[k,_,n],O=p?[S,m,f]:[S,f,m],F=tt({inputs:{x:s},backend:r,attrs:{shape:A}}),P=tt({inputs:{x:t},backend:r,attrs:{shape:O}}),V=r.dataIdMap.get(F.dataId).id,D=r.dataIdMap.get(P.dataId).id,j=u?F.shape[2]:F.shape[1],G=p?P.shape[1]:P.shape[2],X=Math.max(k,S),$=r.makeOutput([X,j,G],F.dtype),le=r.dataIdMap.get($.dataId).id,Q=new Uint8Array(new Int32Array(F.shape).buffer),ce=new Uint8Array(new Int32Array(P.shape).buffer);return xs(V,Q,F.shape.length,D,ce,P.shape.length,u,p,le),r.disposeData(F.dataId),r.disposeData(P.dataId),$.shape=w,$}const Sl={kernelName:Sa,backendName:"wasm",setupFunc:kl,kernelFunc:Il};/**
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
 */function on(a){const{inputs:{x:e},attrs:{begin:r,size:i},backend:s}=a,[t,u]=du(e,r,i),p=hu(e.shape,t,u),h=s.readSync(e.dataId),d=s.makeOutput(u,e.dtype),n=ht(e.shape),f=s.dataIdMap.get(d.dataId);if(p){const y=fu(t,n);return e.dtype==="string"?f.stringBytes=h.slice(y,y+ee(u)):s.typedArrayFromHeap(d).set(h.subarray(y,y+ee(u))),d}if(e.dtype==="string"){const y=Dr(h,t,u,e.shape,e.dtype);return f.stringBytes=y,d}const _=s.typedArrayFromHeap(d),m=e.shape.length;if(m===2)xl(h,n[0],_,t,u);else if(m===3)Ml(h,n[0],n[1],_,t,u);else if(m===4)Al(h,n[0],n[1],n[2],_,t,u);else{const y=Dr(h,t,u,e.shape,e.dtype);_.set(y)}return d}function xl(a,e,r,i,s){let t=0;const u=i[0],p=i[1],h=u+s[0];for(let d=u;d<h;d++){const n=d*e+p;r.set(a.subarray(n,n+s[1]),t),t+=s[1]}}function Ml(a,e,r,i,s,t){let u=0;const p=s[0],h=s[1],d=s[2],n=p+t[0],f=h+t[1];for(let _=p;_<n;_++)for(let m=h;m<f;m++){const y=_*e+m*r+d;i.set(a.subarray(y,y+t[2]),u),u+=t[2]}}function Al(a,e,r,i,s,t,u){let p=0;const h=t[0],d=t[1],n=t[2],f=h+u[0],_=d+u[1],m=n+u[2],y=t[3];for(let b=h;b<f;b++)for(let k=d;k<_;k++)for(let S=n;S<m;S++){const I=b*e+k*r+S*i+y;s.set(a.subarray(I,I+u[3]),p),p+=u[3]}}const Tl={kernelName:Xi,backendName:"wasm",kernelFunc:on};/**
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
 */function Rl(a){const{inputs:e,backend:r,attrs:i}=a,{x:s}=e,{blockShape:t,crops:u}=i,p=t.reduce((S,I)=>S*I),h=fs(s.shape,t,p),d=ms(h.length,t.length),n=gs(s.shape,t,p),f=mu(u,t.length),_=gu(n,u,t.length),m=tt({inputs:{x:s},backend:r,attrs:{shape:h}}),y=Ht({inputs:{x:m},backend:r,attrs:{perm:d}}),b=tt({inputs:{x:y},backend:r,attrs:{shape:n}}),k=on({inputs:{x:b},backend:r,attrs:{begin:f,size:_}});return r.disposeData(m.dataId),r.disposeData(y.dataId),r.disposeData(m.dataId),k}const Fl={kernelName:Yi,backendName:"wasm",kernelFunc:Rl};/**
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
 */function _n(a){const{inputs:{x:e},attrs:{dtype:r},backend:i}=a,s=i.makeOutput(e.shape,r),t=i.typedArrayFromHeap(e);return i.typedArrayFromHeap(s).set(t),s}const El={kernelName:Qi,backendName:"wasm",kernelFunc:_n};/**
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
 */const Nl=Ve(Ji);/**
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
 */let Ms;function Cl(a){Ms=a.wasm.cwrap(xa,null,["number","number","number","number"])}function Ol(a){const{inputs:e,backend:r,attrs:i}=a,{x:s}=e,{clipValueMin:t,clipValueMax:u}=i,p=r.dataIdMap.get(s.dataId).id,h=r.makeOutput(s.shape,s.dtype),d=r.dataIdMap.get(h.dataId).id;return Ms(p,t,u,d),h}const Pl={kernelName:xa,backendName:"wasm",setupFunc:Cl,kernelFunc:Ol};/**
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
 */function As(a){const{inputs:e,backend:r}=a,i=Cn(a.attrs.axis,e[0].shape)[0],s=e.map(m=>m.shape);yu(s,i);let t=nr(e.map(m=>m.shape),i);const u=e.filter(m=>ee(m.shape)>0);if(u.length===1)return qn({inputs:{x:u[0]},backend:r});const p=r.makeOutput(t,e[0].dtype);if(ee(t)===0)return p;if(u[0].dtype==="string"){const m=u.map(w=>{const O=[-1,ee(w.shape.slice(i))];return tt({inputs:{x:w},backend:r,attrs:{shape:O}})}),y=m.map(w=>({vals:r.readSync(w.dataId),shape:w.shape}));t=nr(m.map(w=>w.shape),1);const b=m[0].shape[0]===1,k=bu(y,t,e[0].dtype,b),S=nr(u.map(w=>w.shape),i);p.shape=S;const I=r.dataIdMap.get(p.dataId);return I.stringBytes=_u(k),m.forEach(w=>r.disposeData(w.dataId)),p}const h=ee(u[0].shape.slice(0,i));let d=0;const n=u.map(m=>{const y=ee(m.shape.slice(i));return d+=y,y}),f=u.map(m=>r.typedArrayFromHeap(m)),_=r.typedArrayFromHeap(p);for(let m=0;m<h;m++){let y=m*d;for(let b=0;b<f.length;b++){const k=n[b],S=m*k,I=f[b].subarray(S,S+k);_.set(I,y),y+=k}}return p}const Dl={kernelName:Zi,backendName:"wasm",kernelFunc:As};/**
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
 */let Ts;function Ll(a){Ts=a.wasm.cwrap(Ma,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function Bl(a){const{inputs:e,attrs:r,backend:i}=a,{x:s,filter:t}=e,u=i.dataIdMap.get(s.dataId).id,p=i.dataIdMap.get(t.dataId).id,{strides:h,dilations:d,pad:n,dimRoundingMode:f,dataFormat:_}=r,m=Aa(_),y=On(s.shape,t.shape,h,d,n,f,!1,m),b=y.filterHeight,k=y.filterWidth,S=y.padInfo.top,I=y.padInfo.right,w=y.padInfo.bottom,A=y.padInfo.left,O=y.dilationHeight,F=y.dilationWidth,P=y.strideHeight,V=y.strideWidth,D=y.inChannels,j=y.outChannels,G=y.padInfo.type==="SAME"?1:0;if(y.dataFormat!=="channelsLast")throw new Error(`wasm backend Conv2D does not support dataFormat:'${y.dataFormat}'. Please use 'channelsLast'.`);const X=i.makeOutput(y.outShape,"float32"),$=i.dataIdMap.get(X.dataId).id;return Ts(u,s.shape[0],s.shape[1],s.shape[2],p,b,k,S,I,w,A,G,O,F,P,V,D,j,$),X}const jl={kernelName:Ma,backendName:"wasm",setupFunc:Ll,kernelFunc:Bl};/**
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
 */let Rs;function Vl(a){Rs=a.wasm.cwrap(Ta,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function Wl(a){const{backend:e,inputs:r,attrs:i}=a,{dy:s,filter:t}=r,{strides:u,pad:p,dataFormat:h,dimRoundingMode:d,inputShape:n}=i,f=1,_=Aa(h),m=On(n,t.shape,u,f,p,d,!1,_),{batchSize:y,filterHeight:b,filterWidth:k,inChannels:S,inHeight:I,inWidth:w,outChannels:A,outHeight:O,outWidth:F,strideHeight:P,strideWidth:V}=m,D=b-1-m.padInfo.top,j=k-1-m.padInfo.left,G=m.dataFormat==="channelsLast",X=ht(m.inShape),$=ht(s.shape),[le,Q,ce]=ht(t.shape),ke=X[0],me=G?X[1]:X[2],We=G?X[2]:1,qe=G?1:X[1],Le=$[0],ft=G?$[1]:$[2],Be=G?$[2]:1,Ct=G?1:$[1],xt=e.makeOutput(m.inShape,"float32"),zt=e.dataIdMap.get(xt.dataId).id,Ot=e.dataIdMap.get(s.dataId).id,Fe=e.dataIdMap.get(t.dataId).id;return Rs(Ot,Fe,y,b,k,I,w,S,O,F,A,P,V,D,j,le,Q,ce,ke,me,We,qe,Le,ft,Be,Ct,zt),xt}const Hl={kernelName:Ta,backendName:"wasm",setupFunc:Vl,kernelFunc:Wl};/**
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
 */const Ul=Ve(eo);/**
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
 */const zl=Ve(to);/**
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
 */var yr;(function(a){a[a.bilinear=0]="bilinear",a[a.nearest=1]="nearest"})(yr||(yr={}));let Fs;function $l(a){Fs=a.wasm.cwrap(Ra,null,["number","number","number","number","array","number","number","number","number","number"])}function Gl(a){const{backend:e,inputs:r,attrs:i}=a,{method:s,extrapolationValue:t,cropSize:u}=i,{image:p,boxes:h,boxInd:d}=r,n=h.shape[0],[f,_]=u,m=[n,f,_,p.shape[3]];let y=e.dataIdMap.get(p.dataId),b;p.dtype!=="float32"&&(b=_n({backend:e,inputs:{x:p},attrs:{dtype:"float32"}}),y=e.dataIdMap.get(b.dataId));const k=y.id,S=e.dataIdMap.get(h.dataId).id,I=e.dataIdMap.get(d.dataId).id,w=e.makeOutput(m,"float32"),A=e.dataIdMap.get(w.dataId).id,O=new Uint8Array(new Int32Array(p.shape).buffer);return Fs(k,S,I,n,O,f,_,yr[s],t,A),b!=null&&e.disposeData(b.dataId),w}const ql={kernelName:Ra,backendName:"wasm",setupFunc:$l,kernelFunc:Gl};/**
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
 */let Es;function Kl(a){Es=a.wasm.cwrap(Fa,null,["number","number","number","number","number","number"])}function Xl(a){const{inputs:e,backend:r,attrs:i}=a,{x:s}=e,{axis:t,exclusive:u,reverse:p}=i,h=s.shape.length;et(s.dtype==="float32"||s.dtype==="int32",()=>`cumprod does not support ${s.dtype} tensors in the WASM backend`);const d=Mr([t],h);let n=s;d!==null&&(n=Ht({inputs:{x:s},attrs:{perm:d},backend:r}));const f=yn(1,h)[0];Nt("cumprod",[f],h);const _=r.makeOutput(n.shape,n.dtype),m=n.shape[f],y=r.dataIdMap.get(n.dataId).id,b=r.dataIdMap.get(_.dataId).id;Es(y,u?1:0,p?1:0,m,b,Ee[s.dtype]);let k=_;if(d!==null){const S=Ea(d);k=Ht({inputs:{x:_},attrs:{perm:S},backend:r}),r.disposeData(n.dataId),r.disposeData(_.dataId)}return k}const Yl={kernelName:Fa,backendName:"wasm",setupFunc:Kl,kernelFunc:Xl};/**
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
 */let Ns;function Ql(a){Ns=a.wasm.cwrap(Na,null,["number","number","number","number","number","number"])}function Jl(a){const{inputs:e,backend:r,attrs:i}=a,{x:s}=e,{axis:t,exclusive:u,reverse:p}=i,h=s.shape.length;et(s.dtype==="float32"||s.dtype==="int32",()=>`cumsum does not support ${s.dtype} tensors in the WASM backend`);const d=Mr([t],h);let n=s;d!==null&&(n=Ht({inputs:{x:s},attrs:{perm:d},backend:r}));const f=yn(1,h)[0];Nt("cumsum",[f],h);const _=r.makeOutput(n.shape,n.dtype),m=n.shape[f],y=r.dataIdMap.get(n.dataId).id,b=r.dataIdMap.get(_.dataId).id;Ns(y,u?1:0,p?1:0,m,b,Ee[s.dtype]);let k=_;if(d!==null){const S=Ea(d);k=Ht({inputs:{x:_},attrs:{perm:S},backend:r}),r.disposeData(n.dataId),r.disposeData(_.dataId)}return k}const Zl={kernelName:Na,backendName:"wasm",setupFunc:Ql,kernelFunc:Jl};/**
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
 */let Cs;function ec(a){Cs=a.wasm.cwrap(Ca,null,["number","number","number","array","number","array","array","number","number"])}function tc(a){const{backend:e,inputs:r,attrs:i}=a,{x:s}=r,{blockSize:t,dataFormat:u}=i,p=s.shape[0],h=u==="NHWC"?s.shape[1]:s.shape[2],d=u==="NHWC"?s.shape[2]:s.shape[3],n=u==="NHWC"?s.shape[3]:s.shape[1],f=h*t,_=d*t,m=n/(t*t),y=u==="NHWC"?[p,f,_,m]:[p,m,f,_],b=e.makeOutput(y,"float32"),S=e.dataIdMap.get(s.dataId).id,I=new Uint8Array(new Int32Array(ht(s.shape)).buffer),w=new Uint8Array(new Int32Array(y).buffer),A=new Uint8Array(new Int32Array(ht(y)).buffer),O=e.dataIdMap.get(b.dataId).id;return Cs(S,t,u==="NHWC"?1:0,I,s.shape.length-1,w,A,y.length,O),b}const nc={kernelName:Ca,backendName:"wasm",setupFunc:ec,kernelFunc:tc};/**
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
 */let Os;function rc(a){Os=a.wasm.cwrap(Oa,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function ac(a){const{inputs:e,attrs:r,backend:i}=a,{x:s,filter:t}=e,u=i.dataIdMap.get(s.dataId).id,p=i.dataIdMap.get(t.dataId).id,{strides:h,dilations:d,pad:n,dimRoundingMode:f}=r,_=d??[1,1],m=On(s.shape,t.shape,h,_,n,f,!0),y=m.filterHeight,b=m.filterWidth,k=m.padInfo.top,S=m.padInfo.right,I=m.padInfo.bottom,w=m.padInfo.left,A=m.dilationHeight,O=m.dilationWidth,F=m.strideHeight,P=m.strideWidth,V=m.inChannels,D=m.outChannels,j=m.padInfo.type==="SAME"?1:0;if(m.dataFormat!=="channelsLast")throw new Error(`wasm backend DepthwiseConv2dNative does not support dataFormat:'${m.dataFormat}'. Please use 'channelsLast'.`);const G=i.makeOutput(m.outShape,"float32"),X=i.dataIdMap.get(G.dataId).id;return Os(u,s.shape[0],s.shape[1],s.shape[2],p,y,b,k,S,I,w,j,A,O,F,P,V,D,X),G}const sc={kernelName:Oa,backendName:"wasm",setupFunc:rc,kernelFunc:ac};/**
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
 */const ic=Ve(no);/**
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
 */const oc=!1,uc=Ge(ro,oc,"bool");/**
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
 */const lc=Ve(ao,"float32");/**
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
 */function br(a){const{inputs:e,attrs:r,backend:i}=a,{input:s}=e,{dim:t}=r,u=s.shape.length,p=s.shape.slice();let h=t;return t<0&&(et(-(u+1)<=t,()=>`Axis must be in the interval [${-(u+1)}, ${u}]`),h=u+t+1),p.splice(h,0,1),tt({inputs:{x:s},backend:i,attrs:{shape:p}})}const cc={kernelName:so,backendName:"wasm",kernelFunc:br};/**
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
 */function Ps(a){const{attrs:{shape:e,value:r,dtype:i},backend:s}=a,t=s.makeOutput(e,i);return s.typedArrayFromHeap(t).fill(r),t}const pc={kernelName:io,backendName:"wasm",kernelFunc:Ps};/**
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
 */let Ds;function dc(a){Ds=a.wasm.cwrap(Pa,null,["number","number","number","number","number","number"])}function hc(a){const{inputs:e,backend:r}=a,{image:i}=e,s=r.makeOutput(i.shape,i.dtype),t=r.dataIdMap.get(i.dataId).id,u=r.dataIdMap.get(s.dataId).id,[p,h,d,n]=i.shape;return Ds(t,p,h,d,n,u),s}const fc={kernelName:Pa,backendName:"wasm",kernelFunc:hc,setupFunc:dc};/**
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
 */const mc=Ve(oo);/**
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
 */const gc=Ge(uo);/**
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
 */let Ls;function yc(a){Ls=a.wasm.cwrap(Da,null,["number","number","number","number","number","number","number"])}function bc(a){const{backend:e,inputs:r,attrs:i}=a,{varianceEpsilon:s}=i,{x:t,mean:u,variance:p,offset:h,scale:d}=r,n=e.dataIdMap.get(t.dataId).id,f=e.dataIdMap.get(u.dataId).id,_=e.dataIdMap.get(p.dataId).id,m=h!=null?e.dataIdMap.get(h.dataId).id:0,y=d!=null?e.dataIdMap.get(d.dataId).id:0,b=e.makeOutput(t.shape,t.dtype);if(ee(t.shape)===0)return b;const k=e.dataIdMap.get(b.dataId).id;return Ls(n,f,_,m,y,s,k),b}const _c={kernelName:Da,backendName:"wasm",setupFunc:yc,kernelFunc:bc};/**
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
 */let Bs;function vc(a){Bs=a.wasm.cwrap(La,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function wc(a){const{inputs:e,attrs:r,backend:i}=a,{x:s,filter:t,bias:u,preluActivationWeights:p}=e,{strides:h,pad:d,dilations:n,dataFormat:f,dimRoundingMode:_,activation:m,leakyreluAlpha:y}=r,b=On(s.shape,t.shape,h,n,d,_),k=Fn[m];if(k==null)throw new Error(`${m} activation not yet supported for FusedConv2D in the wasm backend.`);const S=i.dataIdMap.get(s.dataId).id,I=i.dataIdMap.get(t.dataId).id,w=b.outChannels;let A=0;if(u!=null){const Be=i.dataIdMap.get(u.dataId);if(Be.shape.length!==1)throw new Error(`FusedConv2D only supports rank-1 bias but got rank ${Be.shape.length}.`);if(Be.shape[0]!==w)throw new Error(`FusedConv2D bias shape (${Be.shape}) does not match the number of output channels (${w})`);A=Be.id}const O=b.filterHeight,F=b.filterWidth,P=b.padInfo.top,V=b.padInfo.right,D=b.padInfo.bottom,j=b.padInfo.left,G=b.dilationHeight,X=b.dilationWidth,$=b.strideHeight,le=b.strideWidth,Q=b.inChannels,ce=b.padInfo.type==="SAME"?1:0,ke=b.batchSize,me=b.inHeight,We=b.inWidth;if(f!=="NHWC")throw new Error(`wasm backend FusedConv2D does not support dataFormat:'${f}'. Please use 'NHWC'.`);const qe=i.makeOutput(b.outShape,"float32"),Le=i.dataIdMap.get(qe.dataId).id,ft=p==null?0:i.dataIdMap.get(p.dataId).id;return Bs(S,ke,me,We,I,O,F,A,P,V,D,j,ce,G,X,$,le,Q,w,k,ft,y||0,Le),qe}const kc={kernelName:La,backendName:"wasm",setupFunc:vc,kernelFunc:wc};/**
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
 */let js;function Ic(a){js=a.wasm.cwrap(Ba,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function Sc(a){const{inputs:e,attrs:r,backend:i}=a,{x:s,filter:t,bias:u,preluActivationWeights:p}=e,{strides:h,pad:d,dilations:n,dataFormat:f,dimRoundingMode:_,activation:m,leakyreluAlpha:y}=r,b=On(s.shape,t.shape,h,n,d,_,!0),k=Fn[m];if(k==null)throw new Error(`${m} activation not yet supported for FusedDepthwiseConv2D in the wasm backend.`);const S=i.dataIdMap.get(s.dataId).id,I=i.dataIdMap.get(t.dataId).id,w=b.outChannels;let A=0;if(u!=null){const Be=i.dataIdMap.get(u.dataId);if(Be.shape.length!==1)throw new Error(`FusedDepthwiseConv2D only supports rank-1 bias but got rank ${Be.shape.length}.`);if(Be.shape[0]!==w)throw new Error(`FusedDepthwiseConv2D bias shape (${Be.shape}) does not match the number of output channels (${w})`);A=Be.id}const O=b.filterHeight,F=b.filterWidth,P=b.padInfo.top,V=b.padInfo.right,D=b.padInfo.bottom,j=b.padInfo.left,G=b.dilationHeight,X=b.dilationWidth,$=b.strideHeight,le=b.strideWidth,Q=b.inChannels,ce=b.padInfo.type==="SAME"?1:0,ke=b.batchSize,me=b.inHeight,We=b.inWidth;if(f!=="NHWC")throw new Error(`wasm backend FusedDepthwiseConv2D does not support dataFormat:'${f}'. Please use 'NHWC'.`);const qe=i.makeOutput(b.outShape,"float32"),Le=i.dataIdMap.get(qe.dataId).id,ft=p==null?0:i.dataIdMap.get(p.dataId).id;return js(S,ke,me,We,I,O,F,A,P,V,D,j,ce,G,X,$,le,Q,w,k,ft,y||0,Le),qe}const xc={kernelName:Ba,backendName:"wasm",setupFunc:Ic,kernelFunc:Sc};/**
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
 */let Vs;function Mc(a){Vs=a.wasm.cwrap(ja,null,["number","number","number","number","number","number","array","number"])}function Ac(a){const{backend:e,inputs:r}=a,{params:i,indices:s}=r,[t,u,p,h]=vu(i,s),d=e.makeOutput(t,i.dtype);if(u===0)return d;const n=s.shape,f=n[n.length-1],m=e.dataIdMap.get(i.dataId).id,b=e.dataIdMap.get(s.dataId).id,k=new Uint8Array(new Int32Array(h).buffer),S=e.dataIdMap.get(d.dataId).id;return Vs(m,Ee[i.dtype],b,u,f,p,k,S),d}const Tc={kernelName:ja,backendName:"wasm",setupFunc:Mc,kernelFunc:Ac};/**
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
 */let Ws;function Rc(a){Ws=a.wasm.cwrap("Gather",null,["number","number","array","number","number","number","array","number"])}function Fc(a){const{backend:e,inputs:r,attrs:i}=a,{x:s,indices:t}=r,{axis:u,batchDims:p}=i,h=Cn(u,s.shape)[0],d=e.readSync(t.dataId),n=s.shape[h];for(let D=0;D<d.length;++D){const j=d[D];et(j<=n-1&&j>=0,()=>`GatherV2: the index value ${j} is not in [0, ${n-1}]`)}const f=wu(s,t,h,p),_=tt({inputs:{x:s},attrs:{shape:[f.batchSize,f.outerSize,f.dimSize,f.sliceSize]},backend:e}),m=ee(t.shape),y=tt({inputs:{x:t},attrs:{shape:[f.batchSize,m/f.batchSize]},backend:e}),b=[f.batchSize,f.outerSize,m/f.batchSize,f.sliceSize],k=e.makeOutput(b,s.dtype);if(ee(s.shape)===0)return k;const S=_.shape.length-1,w=e.dataIdMap.get(_.dataId).id,O=e.dataIdMap.get(y.dataId).id,F=e.dataIdMap.get(k.dataId).id,P=new Uint8Array(new Int32Array(ht(_.shape)).buffer),V=new Uint8Array(new Int32Array(ht(b)).buffer);return Ws(w,Ee[s.dtype],P,S,O,f.batchSize,V,F),e.disposeData(_.dataId),e.disposeData(y.dataId),k.shape=f.outputShape,k}const Ec={kernelName:lo,backendName:"wasm",setupFunc:Rc,kernelFunc:Fc};/**
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
 */const Nc=!1,Cc=Ge(co,Nc,"bool");/**
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
 */const Oc=!1,Pc=Ge(po,Oc,"bool");/**
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
 */let Hs;function Dc(a){Hs=a.wasm.cwrap(Va,null,["number","number","number","number"])}function Lc(a){const{inputs:{x:e},attrs:{alpha:r},backend:i}=a,s=i.dataIdMap.get(e.dataId).id,t=i.makeOutput(e.shape,"float32");if(ee(e.shape)!==0){const u=i.dataIdMap.get(t.dataId).id;Hs(s,Ee[e.dtype],r,u)}return t}const Bc={kernelName:Va,backendName:"wasm",setupFunc:Dc,kernelFunc:Lc};/**
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
 */const jc=!1,Vc=Ge(ho,jc,"bool");/**
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
 */const Wc=!1,Hc=Ge(fo,Wc,"bool");/**
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
 */const Uc=Ve(mo);/**
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
 */const zc=!1,$c=Ge(go,zc,"bool");/**
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
 */const Gc=Ve(yo);/**
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
 */const qc=!1,Kc=Ge(bo,qc,"bool");/**
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
 */const Xc=!1,Yc=Ge(_o,Xc,"bool");/**
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
 */let Us;function Qc(a){Us=a.wasm.cwrap(Wa,null,["number","number","number","number"])}function Jc(a){const{backend:e,inputs:r,attrs:i}=a,{reductionIndices:s,keepDims:t}=i,{x:u}=r;let h=e.dataIdMap.get(u.dataId).id,d=u;const{transposed:n,axes:f,originalAxes:_,inputWasTransposed:m}=Ut(u,s,e);if(m){const w=e.dataIdMap.get(n.dataId).id;d=n,h=w}const y=d.shape.length;Nt("max",f,y);const[b,k]=un(d.shape,f),S=ee(k),I=e.makeOutput(b,u.dtype);if(ee(d.shape)!==0){const w=e.dataIdMap.get(I.dataId).id;Us(h,Ee[u.dtype],S,w)}if(m&&e.disposeData(n.dataId),t){const w=ln(I.shape,_);I.shape=w}return I}const Zc={kernelName:Wa,backendName:"wasm",setupFunc:Qc,kernelFunc:Jc};/**
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
 */const ep=Ge(vo);/**
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
 */let zs;function tp(a){zs=a.wasm.cwrap(Ha,null,["number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number","number"])}function np(a){const{inputs:e,attrs:r,backend:i}=a,s=e.x,t=i.dataIdMap.get(s.dataId).id;et(s.dtype==="float32",()=>`Error in MaxPool: only float32 input is supported. Got ${s.dtype}.`);const{filterSize:u,strides:p,pad:h,dimRoundingMode:d}=r,n=Ia(s.shape,u,p,1,h,d),f=n.filterHeight,_=n.filterWidth,m=n.padInfo.top,y=n.padInfo.right,b=n.padInfo.bottom,k=n.padInfo.left,S=n.dilationHeight,I=n.dilationWidth,w=n.strideHeight,A=n.strideWidth,O=n.inChannels,F=n.outChannels;if(n.dataFormat!=="channelsLast")throw new Error(`wasm backend does not support dataFormat:'${n.dataFormat}'. Please use 'channelsLast'.`);const P=i.makeOutput(n.outShape,"float32"),V=i.dataIdMap.get(P.dataId).id;return zs(t,s.shape[0],s.shape[1],s.shape[2],f,_,m,y,b,k,S,I,w,A,O,F,V),P}const rp={kernelName:Ha,backendName:"wasm",setupFunc:tp,kernelFunc:np};/**
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
 */let $s;function ap(a){$s=a.wasm.cwrap(Ua,null,["number, number, number"])}function sp(a){const{backend:e,inputs:r,attrs:i}=a,{axis:s,keepDims:t}=i,{x:u}=r,p=e.dataIdMap.get(u.dataId).id;let h=p,d=u;const{transposed:n,axes:f,originalAxes:_,inputWasTransposed:m}=Ut(u,s,e);let y=f;if(m){const A=e.dataIdMap.get(n.dataId).id;A!==p&&(d=n,h=A,y=yn(y.length,d.shape.length))}Nt("mean",y,d.shape.length);const[b,k]=un(d.shape,y),S=ee(k);let I=d;d.dtype!=="float32"&&(I=_n({backend:e,inputs:{x:d},attrs:{dtype:"float32"}}),h=e.dataIdMap.get(I.dataId).id);const w=e.makeOutput(b,"float32");if(ee(d.shape)!==0){const A=e.dataIdMap.get(w.dataId).id;$s(h,S,A)}if(m&&e.disposeData(n.dataId),t){const A=ln(w.shape,_);w.shape=A}return d.dtype!=="float32"&&e.disposeData(I.dataId),w}const ip={kernelName:Ua,backendName:"wasm",setupFunc:ap,kernelFunc:sp};/**
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
 */let Gs;function op(a){Gs=a.wasm.cwrap(za,null,["number","number","number","number"])}function up(a){const{backend:e,inputs:r,attrs:i}=a,{axis:s,keepDims:t}=i,{x:u}=r,p=e.dataIdMap.get(u.dataId).id;let h=p,d=u;const{transposed:n,axes:f,originalAxes:_,inputWasTransposed:m}=Ut(u,s,e);if(m){const w=e.dataIdMap.get(n.dataId).id;w!==p&&(d=n,h=w)}const y=d.shape.length;Nt("min",f,y);const[b,k]=un(d.shape,f),S=ee(k),I=e.makeOutput(b,d.dtype);if(ee(d.shape)!==0){const w=e.dataIdMap.get(I.dataId).id;Gs(h,Ee[u.dtype],S,w)}if(m&&e.disposeData(n.dataId),t){const w=ln(I.shape,_);I.shape=w}return I}const lp={kernelName:za,backendName:"wasm",setupFunc:op,kernelFunc:up};/**
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
 */const cp=Ge(wo);/**
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
 */var _r;(function(a){a[a.reflect=0]="reflect",a[a.symmetric=1]="symmetric"})(_r||(_r={}));let qs;function pp(a){qs=a.wasm.cwrap($a,null,["number","array","number","number","array","array","number","number"])}function dp(a){const{inputs:{x:e},backend:r,attrs:{paddings:i,mode:s}}=a,t=i.map((y,b)=>y[0]+e.shape[b]+y[1]),u=r.dataIdMap.get(e.dataId).id,p=r.makeOutput(t,e.dtype),h=r.dataIdMap.get(p.dataId).id,d=new Uint8Array(new Int32Array(e.shape).buffer),n=i.map(y=>y[0]),f=i.map(y=>y[1]),_=new Uint8Array(new Int32Array(n).buffer),m=new Uint8Array(new Int32Array(f).buffer);return qs(u,d,e.shape.length,Ee[e.dtype],_,m,_r[s],h),p}const hp={kernelName:$a,backendName:"wasm",kernelFunc:dp,setupFunc:pp};/**
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
 */const fp=Ge(ko);/**
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
 */const mp=Ve(Io);/**
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
 */function Rr(a,e){const r=new Int32Array(a.wasm.HEAPU8.buffer,e,4),i=r[0],s=r[1],t=r[2],u=r[3];return a.wasm._free(e),{pSelectedIndices:i,selectedSize:s,pSelectedScores:t,pValidOutputs:u}}/**
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
 */let Ks;function gp(a){Ks=a.wasm.cwrap(Ga,"number",["number","number","number","number","number"])}function yp(a){const{backend:e,inputs:r,attrs:i}=a,{iouThreshold:s,maxOutputSize:t,scoreThreshold:u}=i,{boxes:p,scores:h}=r,d=e.dataIdMap.get(p.dataId).id,n=e.dataIdMap.get(h.dataId).id,f=Ks(d,n,t,s,u),{pSelectedIndices:_,selectedSize:m,pSelectedScores:y,pValidOutputs:b}=Rr(e,f);return e.wasm._free(y),e.wasm._free(b),e.makeOutput([m],"int32",_)}const bp={kernelName:Ga,backendName:"wasm",setupFunc:gp,kernelFunc:yp};/**
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
 */let Xs;function _p(a){Xs=a.wasm.cwrap(qa,"number",["number","number","number","number","number","bool"])}function vp(a){const{backend:e,inputs:r,attrs:i}=a,{iouThreshold:s,maxOutputSize:t,scoreThreshold:u,padToMaxOutputSize:p}=i,{boxes:h,scores:d}=r,n=e.dataIdMap.get(h.dataId).id,f=e.dataIdMap.get(d.dataId).id,_=Xs(n,f,t,s,u,p),{pSelectedIndices:m,selectedSize:y,pSelectedScores:b,pValidOutputs:k}=Rr(e,_);e.wasm._free(b);const S=e.makeOutput([y],"int32",m),I=e.makeOutput([],"int32",k);return[S,I]}const wp={kernelName:qa,backendName:"wasm",setupFunc:_p,kernelFunc:vp};/**
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
 */let Ys;function kp(a){Ys=a.wasm.cwrap(Ka,"number",["number","number","number","number","number","number"])}function Ip(a){const{backend:e,inputs:r,attrs:i}=a,{iouThreshold:s,maxOutputSize:t,scoreThreshold:u,softNmsSigma:p}=i,{boxes:h,scores:d}=r,n=e.dataIdMap.get(h.dataId).id,f=e.dataIdMap.get(d.dataId).id,_=Ys(n,f,t,s,u,p),{pSelectedIndices:m,selectedSize:y,pSelectedScores:b,pValidOutputs:k}=Rr(e,_);e.wasm._free(k);const S=e.makeOutput([y],"int32",m),I=e.makeOutput([y],"float32",b);return[S,I]}const Sp={kernelName:Ka,backendName:"wasm",setupFunc:kp,kernelFunc:Ip};/**
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
 */const xp=!1,Mp=Ge(So,xp,"bool");/**
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
 */let Qs;function Ap(a){Qs=a.wasm.cwrap(Xa,null,["number","number","number","number","number"])}function Tp(a){const{inputs:e,backend:r,attrs:i}=a,{indices:s}=e,{dtype:t,depth:u,onValue:p,offValue:h}=i,d=r.makeOutput([...s.shape,u],t),n=r.dataIdMap.get(d.dataId).id,_=r.dataIdMap.get(s.dataId).id;return Qs(_,u,p,h,n),d}const Rp={kernelName:Xa,backendName:"wasm",setupFunc:Ap,kernelFunc:Tp};/**
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
 */function Fp(a){const{inputs:{x:e},backend:r}=a,i=r.makeOutput(e.shape,e.dtype);return r.typedArrayFromHeap(i).fill(1),i}const Ep={kernelName:xo,backendName:"wasm",kernelFunc:Fp};/**
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
 */function Np(a){const{inputs:e,backend:r,attrs:i}=a,{axis:s}=i;if(e.length===1)return br({inputs:{input:e[0]},backend:r,attrs:{dim:s}});const t=e[0].shape,u=e[0].dtype;e.forEach(n=>{Ao(t,n.shape,"All tensors passed to stack must have matching shapes"),et(u===n.dtype,()=>"All tensors passed to stack must have matching dtypes")});const p=[],h=e.map(n=>{const f=br({inputs:{input:n},backend:r,attrs:{dim:s}});return p.push(f),f}),d=As({inputs:h,backend:r,attrs:{axis:s}});return p.forEach(n=>r.disposeData(n.dataId)),d}const Cp={kernelName:Mo,backendName:"wasm",kernelFunc:Np};/**
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
 */let Js;function Op(a){Js=a.wasm.cwrap(Ya,null,["number","array","number","number","array","array","number","number"])}function Pp(a){const{inputs:{x:e},backend:r,attrs:{paddings:i,constantValue:s}}=a,t=i.map((b,k)=>b[0]+e.shape[k]+b[1]);if(ee(e.shape)===0)return Ps({backend:r,attrs:{shape:t,value:s,dtype:e.dtype}});const u=r.dataIdMap.get(e.dataId).id,p=r.makeOutput(t,e.dtype),d=r.dataIdMap.get(p.dataId).id,n=new Uint8Array(new Int32Array(e.shape).buffer),f=i.map(b=>b[0]),_=i.map(b=>b[1]),m=new Uint8Array(new Int32Array(f).buffer),y=new Uint8Array(new Int32Array(_).buffer);return Js(u,n,e.shape.length,Ee[e.dtype],m,y,s,d),p}const Zs={kernelName:Ya,backendName:"wasm",kernelFunc:Pp,setupFunc:Op};/**
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
 */const Dp=Ge(To);/**
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
 */let ei;function Lp(a){ei=a.wasm.cwrap(Qa,null,["number","number","number"])}function Bp(a){const{inputs:e,backend:r}=a,{x:i,alpha:s}=e,t=r.dataIdMap.get(i.dataId).id,u=r.dataIdMap.get(s.dataId).id;let p=t;const h=i;let d=h;h.dtype!=="float32"&&(d=_n({backend:r,inputs:{x:i},attrs:{dtype:"float32"}}),p=r.dataIdMap.get(d.dataId).id);const n=r.makeOutput(i.shape,"float32"),f=r.dataIdMap.get(n.dataId).id;return ei(p,u,f),h.dtype!=="float32"&&r.disposeData(d.dataId),n}const jp={kernelName:Qa,backendName:"wasm",setupFunc:Lp,kernelFunc:Bp};/**
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
 */let ti;function Vp(a){ti=a.wasm.cwrap(Ja,null,["number","number","number","number"])}function Wp(a){const{backend:e,inputs:r,attrs:i}=a,{axis:s,keepDims:t}=i,{x:u}=r,p=e.dataIdMap.get(u.dataId).id;let h=p,d=u;const{transposed:n,axes:f,originalAxes:_,inputWasTransposed:m}=Ut(u,s,e);let y=f;if(m){const w=e.dataIdMap.get(n.dataId).id;w!==p&&(d=n,h=w,y=yn(y.length,d.shape.length))}Nt("prod",y,d.shape.length);const[b,k]=un(d.shape,y),S=ee(k),I=e.makeOutput(b,d.dtype);if(ee(d.shape)!==0){const w=e.dataIdMap.get(I.dataId).id;ti(h,S,Ee[I.dtype],w)}if(m&&e.disposeData(n.dataId),t){const w=ln(I.shape,_);I.shape=w}return I}const Hp={kernelName:Ja,backendName:"wasm",setupFunc:Vp,kernelFunc:Wp};/**
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
 */const Up=a=>{const{backend:e,attrs:r}=a,{start:i,stop:s,step:t,dtype:u}=r,p=ku(i,s,t,u),h=e.makeOutput([p.length],u);return e.typedArrayFromHeap(h).set(p),h},zp={kernelName:Ro,backendName:"wasm",kernelFunc:Up};/**
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
 */const $p=Ge(Fo);/**
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
 */const Gp=Ve(Eo);/**
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
 */const qp=Ve(No);/**
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
 */let ni;function Kp(a){ni=a.wasm.cwrap(Za,null,["number","number","number","number","number","number","number","number","number","number"])}function Xp(a){const{backend:e,inputs:r,attrs:i}=a,{images:s}=r,{alignCorners:t,halfPixelCenters:u,size:p}=i,[h,d]=p,[n,f,_,m]=s.shape,y=[n,h,d,m];let b=e.dataIdMap.get(s.dataId),k;b.dtype!=="float32"&&(k=_n({backend:e,inputs:{x:s},attrs:{dtype:"float32"}}),b=e.dataIdMap.get(k.dataId));const S=b.id,I=e.makeOutput(y,"float32");if(ee(s.shape)===0)return I;const w=e.dataIdMap.get(I.dataId).id;return ni(S,n,f,_,m,h,d,t?1:0,u?1:0,w),k!=null&&e.disposeData(k.dataId),I}const Yp={kernelName:Za,backendName:"wasm",setupFunc:Kp,kernelFunc:Xp};/**
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
 */let ri;function Qp(a){ri=a.wasm.cwrap(es,null,["number","number","number","number","number","number","number","number","number","number"])}function Jp(a){const{backend:e,inputs:r,attrs:i}=a,{images:s}=r,{alignCorners:t,halfPixelCenters:u,size:p}=i,[h,d]=p,[n,f,_,m]=s.shape,y=[n,h,d,m],b=e.makeOutput(y,"float32");if(ee(s.shape)===0)return b;let k=e.dataIdMap.get(s.dataId),S;k.dtype!=="float32"&&(S=_n({backend:e,inputs:{x:s},attrs:{dtype:"float32"}}),k=e.dataIdMap.get(S.dataId));const I=k.id,w=e.dataIdMap.get(b.dataId).id;return ri(I,n,f,_,m,h,d,t?1:0,u?1:0,w),S!=null&&e.disposeData(S.dataId),b}const Zp={kernelName:es,backendName:"wasm",setupFunc:Qp,kernelFunc:Jp};/**
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
 */let ai;function ed(a){ai=a.wasm.cwrap(ts,null,["number","array","number","array","number","number"])}function td(a){const{inputs:e,backend:r,attrs:i}=a,{x:s}=e,{dims:t}=i,u=Cn(t,s.shape);if(s.shape.length===0)return qn({inputs:{x:s},backend:r});const p=r.makeOutput(s.shape,s.dtype),h=r.dataIdMap.get(s.dataId).id,d=r.dataIdMap.get(p.dataId).id,n=new Uint8Array(new Int32Array(u).buffer),f=new Uint8Array(new Int32Array(s.shape).buffer);ai(h,n,u.length,f,s.shape.length,d);const _=tt({inputs:{x:p},attrs:{shape:s.shape},backend:r});return r.disposeData(p.dataId),_}const nd={kernelName:ts,backendName:"wasm",kernelFunc:td,setupFunc:ed};/**
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
 */let si;function rd(a){si=a.wasm.cwrap(ns,null,["number","number","number","number","number","number","number","number","array","number","number"])}function ad(a){const{inputs:e,backend:r,attrs:i}=a,{image:s}=e,{radians:t,fillValue:u,center:p}=i,h=r.makeOutput(s.shape,s.dtype),d=r.dataIdMap.get(s.dataId).id,n=r.dataIdMap.get(h.dataId).id,[f,_,m,y]=s.shape,[b,k]=Iu(p,_,m),S=u===0,I=255,w=typeof u=="number"?[u,u,u,S?0:I]:[...u,I],A=new Uint8Array(new Int32Array(w).buffer);return si(d,f,_,m,y,t,b,k,A,w.length,n),h}const sd={kernelName:ns,backendName:"wasm",kernelFunc:ad,setupFunc:rd};/**
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
 */const id=Ve(Co);/**
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
 */const od=Ve(Oo);/**
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
 */let ii;function ud(a){ii=a.wasm.cwrap(rs,null,["number","number","number","number","number","number","array","number","number"])}function ld(a){const{backend:e,inputs:r,attrs:i}=a,{indices:s,updates:t}=r,{shape:u}=i,p=e.makeOutput(u,t.dtype);if(ee(u)===0)return p;const{sliceRank:h,numUpdates:d,sliceSize:n,strides:f,outputSize:_}=Po(t,s,u),y=e.dataIdMap.get(s.dataId).id,k=e.dataIdMap.get(t.dataId).id,S=new Uint8Array(new Int32Array(f).buffer),I=e.dataIdMap.get(p.dataId).id;return ii(y,k,Ee[t.dtype],h,d,n,S,_,I),p}const cd={kernelName:rs,backendName:"wasm",setupFunc:ud,kernelFunc:ld};/**
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
 */let oi;function pd(a){oi=a.wasm.cwrap("SelectV2",null,["number","number","number","number","number"])}function dd(a){const{inputs:e,backend:r}=a,{condition:i,t:s,e:t}=e,u=r.dataIdMap.get(i.dataId).id,p=r.dataIdMap.get(s.dataId).id,h=r.dataIdMap.get(t.dataId).id,d=r.makeOutput(s.shape,s.dtype),n=r.dataIdMap.get(d.dataId).id,f=i.shape.length,_=s.shape.length,m=f===0||f>1||_===1?1:ee(s.shape.slice(1));return oi(u,p,h,m,n),d}const hd={kernelName:Do,backendName:"wasm",kernelFunc:dd,setupFunc:pd};/**
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
 */let ui;function fd(a){ui=a.wasm.cwrap(Lo,null,["number","number"])}function md(a){const{backend:e,inputs:{x:r}}=a,i=e.dataIdMap.get(r.dataId).id,s=e.makeOutput(r.shape,r.dtype),t=e.dataIdMap.get(s.dataId).id;return ee(s.shape)===0||ui(i,t),s}const gd={kernelName:"Sigmoid",backendName:"wasm",setupFunc:fd,kernelFunc:md};/**
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
 */const yd=Ve(Bo);/**
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
 */let li;function bd(a){li=a.wasm.cwrap(as,null,["number","number","number","number"])}function _d(a){const{backend:e,inputs:{logits:r},attrs:{dim:i}}=a,s=e.dataIdMap.get(r.dataId).id,t=e.makeOutput(r.shape,r.dtype),u=e.dataIdMap.get(t.dataId).id,p=r.shape[i],h=ee(r.shape)/p;return ee(t.shape)===0||li(s,u,p,h),t}const vd={kernelName:as,backendName:"wasm",setupFunc:bd,kernelFunc:_d};/**
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
 */function wd(a){const{inputs:e,backend:r,attrs:i}=a,{x:s}=e,{blockShape:t,paddings:u}=i,p=ee(t),h=[[0,0]];h.push(...u);for(let F=1+t.length;F<s.shape.length;++F)h.push([0,0]);const d=Zs.kernelFunc({inputs:{x:s},backend:r,attrs:{paddings:h,constantValue:0}}),n=fs(d.shape,t,p,!1),f=ms(n.length,t.length,!1),_=gs(d.shape,t,p,!1),b=tt({inputs:{x:d},backend:r,attrs:{shape:n}}),I=Ht({inputs:{x:b},backend:r,attrs:{perm:f}}),O=tt({inputs:{x:I},backend:r,attrs:{shape:_}});return r.disposeData(d.dataId),r.disposeData(b.dataId),r.disposeData(I.dataId),O}const kd={kernelName:jo,backendName:"wasm",kernelFunc:wd};/**
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
 */let ci;function Id(a){ci=a.wasm.cwrap("SparseFillEmptyRows","number",["number","number","number","number","number","number","number","number","number","number","number","number"])}function Sd(a){const{backend:e,inputs:r}=a,{indices:i,values:s,denseShape:t,defaultValue:u}=r,p=i.shape[0],h=i.shape[1],d=e.readSync(t.dataId)[0],n=[p+d,h],f=e.dataIdMap.get(i.dataId).id,_=e.dataIdMap.get(s.dataId).id,m=e.dataIdMap.get(u.dataId).id,y=e.makeOutput(n,i.dtype),b=e.dataIdMap.get(y.dataId).id,k=e.makeOutput(n.slice(0,1),s.dtype),S=e.dataIdMap.get(k.dataId).id,I=e.makeOutput([d],"bool"),w=e.dataIdMap.get(I.dataId).id,A=e.makeOutput([p],i.dtype),O=e.dataIdMap.get(A.dataId).id,F=e.makeOutput([4],"int32"),P=e.dataIdMap.get(F.dataId).id,V=ci(f,_,Ee[s.dtype],p,d,h,m,b,S,w,O,P),D=e.readSync(F.dataId);let j;switch(D[0]){case 1:{j=Mu(D[1]);break}case 2:{j=xu(D[1],D[2]);break}case 3:j=Su(D[1],D[2],D[3]);break;default:j=""}if(e.disposeData(F.dataId),j)throw e.disposeData(y.dataId),e.disposeData(k.dataId),e.disposeData(I.dataId),e.disposeData(A.dataId),new Error(j);let G=y,X=k;return V!==n[0]&&(G=on({inputs:{x:y},attrs:{begin:0,size:[V,h]},backend:e}),X=on({inputs:{x:k},attrs:{begin:0,size:V},backend:e}),e.disposeData(y.dataId),e.disposeData(k.dataId)),[G,X,I,A]}const xd={kernelName:Vo,backendName:"wasm",setupFunc:Id,kernelFunc:Sd};/**
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
 */let pi;function Md(a){pi=a.wasm.cwrap(ss,null,["number","number","number","number","number","number","number"])}function Ad(a){const{backend:e,inputs:r}=a,{inputIndices:i,inputShape:s,newShape:t}=r;if(i.shape.length!==2)throw new Error(`Input indices should be a matrix but received shape
        ${i.shape}`);if(s.shape.length!==1)throw new Error(`Input shape should be a vector but received shape
        ${s.shape}`);if(t.shape.length!==1)throw new Error(`Target shape should be a vector but received shape ${t.shape}`);const u=e.dataIdMap.get(i.dataId).id,p=e.dataIdMap.get(s.dataId).id,h=e.dataIdMap.get(t.dataId).id,d=i.shape[0],n=ee(t.shape),f=e.makeOutput([d,n],i.dtype),_=e.dataIdMap.get(f.dataId).id,m=e.makeOutput([n],t.dtype),y=e.dataIdMap.get(m.dataId).id,b=e.makeOutput([3],"int32"),k=e.dataIdMap.get(b.dataId).id;pi(u,p,h,d,_,y,k);const S=e.readSync(b.dataId);let I;switch(S[0]){case 0:{I=Fu(S[1],S[2]);break}case 1:{I=Ru(S[1],S[2]);break}case 2:I=Eu();break;case 3:{const w=Array.from(e.readSync(s.dataId)),A=Array.from(e.readSync(m.dataId));I=Tu(w,A);break}case 4:{const w=Array.from(e.readSync(s.dataId)),A=Array.from(e.readSync(m.dataId));I=Au(w,A);break}default:I=""}if(e.disposeData(b.dataId),I)throw e.disposeData(f.dataId),e.disposeData(m.dataId),new Error(I);return[f,m]}const Td={kernelName:ss,backendName:"wasm",setupFunc:Md,kernelFunc:Ad};/**
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
 */let di;function hi(a){di=a.wasm.cwrap("SparseSegmentReduction",null,["number","number","number","number","number","number","number","number","number"])}function fi(a,e){const{backend:r,inputs:i}=a,{data:s,indices:t,segmentIds:u}=i,p=t.shape[0],h=r.readSync(u.dataId,p-1,p)[0],n=p>0?h+1:0;if(n<0)throw new Error(Lr());const f=s.shape.slice();f[0]=n;const _=r.dataIdMap.get(s.dataId).id,m=r.dataIdMap.get(t.dataId).id,y=r.dataIdMap.get(u.dataId).id,b=r.makeOutput(f,s.dtype),k=r.dataIdMap.get(b.dataId).id,S=r.makeOutput([4],"int32"),I=r.dataIdMap.get(S.dataId).id;di(_,Ee[s.dtype],s.shape[0],m,y,k,I,e,0);const w=r.readSync(S.dataId);let A;switch(w[0]){case 0:{A=Lr();break}case 1:{A=Ou();break}case 2:A=Cu(w[1],w[2]);break;case 3:A=Nu(w[1],w[2],w[3]);break;default:A=""}if(r.disposeData(S.dataId),A)throw r.disposeData(b.dataId),new Error(A);return b}/**
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
 */function Rd(a){return fi(a,!0)}const Fd={kernelName:Wo,backendName:"wasm",setupFunc:hi,kernelFunc:Rd};/**
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
 */function Ed(a){return fi(a,!1)}const Nd={kernelName:Ho,backendName:"wasm",setupFunc:hi,kernelFunc:Ed};/**
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
 */function Cd(a){const{inputs:e,attrs:r,backend:i}=a,{x:s}=e,{numOrSizeSplits:t,axis:u}=r,p=Cn(u,s.shape)[0],h=Pu(s,t,p),d=new Array(s.shape.length).fill(0),n=s.shape.slice();return h.map(f=>{const _=[...n];_[p]=f;const m=on({inputs:{x:s},attrs:{begin:d,size:_},backend:i});return d[p]+=f,m})}const Od={kernelName:Uo,backendName:"wasm",kernelFunc:Cd};/**
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
 */const Pd=Ve(zo);/**
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
 */const Dd=Ve($o);/**
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
 */const Ld=Ge(Go);/**
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
 */let mi;function Bd(a){mi=a.wasm.cwrap(is,null,["number","number","number","number"])}function jd(a){const{backend:e,inputs:r,attrs:i}=a,{alpha:s}=i,{x:t}=r,u=e.dataIdMap.get(t.dataId).id,p=e.makeOutput(t.shape,t.dtype),h=e.dataIdMap.get(p.dataId).id;return mi(u,s,Ee[t.dtype],h),p}const Vd={kernelName:is,backendName:"wasm",setupFunc:Bd,kernelFunc:jd};/**
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
 */let gi;function Wd(a){gi=a.wasm.cwrap(os,null,["number","array","number","array","array","array","array","array","number","number"])}function Hd(a){const{backend:e,inputs:r,attrs:i}=a,{x:s}=r,{begin:t,end:u,strides:p,beginMask:h,endMask:d,ellipsisMask:n,newAxisMask:f,shrinkAxisMask:_}=i,{finalShapeSparse:m,finalShape:y,isIdentity:b,sliceDim0:k,isSimpleSlice:S,begin:I,end:w,strides:A}=Du(s.shape,t,u,p,h,d,n,f,_);let O;if(b)O=tt({inputs:{x:s},backend:e,attrs:{shape:y}});else if(k||S){et(s.shape.length>=1,()=>`Input must have rank at least 1, got: ${s.shape.length}`);const F=Lu(I,w,A),P=on({inputs:{x:s},backend:e,attrs:{begin:I,size:F}});O=tt({inputs:{x:P},backend:e,attrs:{shape:y}}),e.disposeData(P.dataId)}else{const F=e.makeOutput(m,"float32"),P=e.dataIdMap.get(s.dataId).id,V=new Uint8Array(new Int32Array(ht(s.shape)).buffer),D=new Uint8Array(new Int32Array(I).buffer),j=new Uint8Array(new Int32Array(w).buffer),G=new Uint8Array(new Int32Array(A).buffer),X=new Uint8Array(new Int32Array(m).buffer),$=new Uint8Array(new Int32Array(ht(m)).buffer),le=e.dataIdMap.get(F.dataId).id;gi(P,V,s.shape.length,D,j,G,X,$,m.length,le),O=tt({inputs:{x:F},backend:e,attrs:{shape:y}}),e.disposeData(F.dataId)}return O}const Ud={kernelName:os,backendName:"wasm",setupFunc:Wd,kernelFunc:Hd};/**
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
 */function zd(a){const{backend:e,inputs:r,attrs:i}=a,{data:s,dataSplits:t}=r,{separator:u,nGramWidths:p,leftPad:h,rightPad:d,padWidth:n,preserveShortSequences:f}=i,_=e.readSync(s.dataId),m=e.readSync(t.dataId),[y,b]=Bu(_,m,u,p,h,d,n,f),k=e.makeOutput([y.length],"string"),S=e.dataIdMap.get(k.dataId);S.stringBytes=y;const I=e.makeOutput(t.shape,"int32");return e.typedArrayFromHeap(I).set(b),[k,I]}const $d={kernelName:qo,backendName:"wasm",kernelFunc:zd};/**
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
 */function Gd(a){const{backend:e,inputs:r,attrs:i}=a,{input:s,delimiter:t}=r,{skipEmpty:u}=i,p=e.readSync(s.dataId),h=e.readSync(t.dataId),[d,n,f]=ju(p,h[0],u),_=n.length,m=e.makeOutput([_,2],"int32");e.typedArrayFromHeap(m).set(d);const b=e.makeOutput([_],"string"),k=e.dataIdMap.get(b.dataId);k.stringBytes=n;const S=e.makeOutput([2],"int32");return e.typedArrayFromHeap(S).set(f),[m,b,S]}const qd={kernelName:Ko,backendName:"wasm",kernelFunc:Gd};/**
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
 */function Kd(a){const{backend:e,inputs:r,attrs:i}=a,{input:s}=r,{numBuckets:t}=i,u=e.readSync(s.dataId),p=Vu(u,t),h=e.makeOutput(s.shape,"int32");return e.typedArrayFromHeap(h).set(p),h}const Xd={kernelName:Xo,backendName:"wasm",kernelFunc:Kd};/**
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
 */const Yd=Ge(Yo);/**
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
 */let yi;function Qd(a){yi=a.wasm.cwrap(us,null,["number","number","number","number"])}function Jd(a){const{backend:e,inputs:r,attrs:i}=a,{axis:s,keepDims:t}=i,{x:u}=r,p=e.dataIdMap.get(u.dataId).id;let h=p,d=u;const{transposed:n,axes:f,originalAxes:_,inputWasTransposed:m}=Ut(u,s,e);let y=f;if(m){const w=e.dataIdMap.get(n.dataId).id;w!==p&&(d=n,h=w,y=yn(y.length,d.shape.length))}Nt("sum",y,d.shape.length);const[b,k]=un(d.shape,y),S=ee(k),I=e.makeOutput(b,d.dtype);if(ee(d.shape)!==0){const w=e.dataIdMap.get(I.dataId).id;yi(h,S,Ee[I.dtype],w)}if(m&&e.disposeData(n.dataId),t){const w=ln(I.shape,_);I.shape=w}return I}const Zd={kernelName:us,backendName:"wasm",setupFunc:Qd,kernelFunc:Jd};/**
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
 */const eh=Ve(Qo);/**
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
 */const th=Ve(Jo);/**
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
 */let bi;function nh(a){bi=a.wasm.cwrap(ls,null,["number","array","number","array","number","number"])}function rh(a){const{inputs:e,backend:r,attrs:i}=a,{x:s}=e,t=r.dataIdMap.get(s.dataId).id,{reps:u}=i,p=new Array(s.shape.length);for(let _=0;_<p.length;_++)p[_]=s.shape[_]*u[_];const h=new Uint8Array(new Int32Array(s.shape).buffer),d=new Uint8Array(new Int32Array(p).buffer),n=r.makeOutput(p,s.dtype),f=r.dataIdMap.get(n.dataId).id;return bi(t,h,s.shape.length,d,p.length,Ee[n.dtype],f),n}const ah={kernelName:ls,backendName:"wasm",setupFunc:nh,kernelFunc:rh};/**
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
 */let _i;function sh(a){_i=a.wasm.cwrap(cs,null,["number","array","number","number","number","bool","number","number"])}const ih=({inputs:a,backend:e,attrs:r})=>{const{x:i}=a,{k:s,sorted:t}=r,u=e.dataIdMap.get(i.dataId).id,p=new Uint8Array(new Int32Array(i.shape).buffer),h=i.shape.slice();h[h.length-1]=s;const d=e.makeOutput(h,i.dtype),n=e.dataIdMap.get(d.dataId).id,f=e.makeOutput(h,"int32"),_=e.dataIdMap.get(f.dataId).id;return _i(u,p,i.shape.length,Ee[i.dtype],s,t,n,_),[d,f]},oh={kernelName:cs,backendName:"wasm",setupFunc:sh,kernelFunc:ih};/**
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
 */let vi;function uh(a){vi=a.wasm.cwrap(ps,null,["number","number","bool","number","number","number","number","number","number","array","number","array","number","number","number","number","number"])}function lh(a){const{backend:e,inputs:r,attrs:i}=a,{image:s,transforms:t}=r,{interpolation:u,fillMode:p,fillValue:h,outputShape:d}=i,[n,f,_,m]=s.shape,[y,b]=d??[f,_],k=[n,y,b,m],S=new Uint8Array(new Int32Array(ht(s.shape)).buffer),I=new Uint8Array(new Int32Array(ht(k)).buffer),w=e.makeOutput(k,s.dtype),A=e.dataIdMap.get(w.dataId).id,F=e.dataIdMap.get(s.dataId).id,V=e.dataIdMap.get(t.dataId).id,D=u==="nearest"?1:2;let j;switch(p){case"constant":j=1;break;case"reflect":j=2;break;case"wrap":j=3;break;case"nearest":j=4;break;default:j=1;break}return vi(F,V,t.shape[0]>1,n,y,b,m,_,f,S,s.shape.length-1,I,k.length-1,D,j,h,A),w}const ch={kernelName:ps,backendName:"wasm",setupFunc:uh,kernelFunc:lh};/**
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
 */function ph(a){const{inputs:e,backend:r,attrs:i}=a,{value:s}=e;let{axis:t}=i;t<0&&(t+=s.shape.length);const u=s.shape[t],p=s.shape.length,h=new Array(p-1);let d=0;for(let m=0;m<p;m++)m!==t&&(h[d++]=s.shape[m]);const n=new Array(u),f=new Array(p).fill(0),_=s.shape.slice();_[t]=1;for(let m=0;m<n.length;m++)f[t]=m,n[m]=on({inputs:{x:s},attrs:{begin:f,size:_},backend:r});return n.map(({dataId:m,dtype:y})=>({dataId:m,dtype:y,shape:h}))}const dh={kernelName:Zo,backendName:"wasm",kernelFunc:ph};/**
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
 */function hh(a){const{inputs:{x:e},backend:r}=a,i=r.makeOutput(e.shape,e.dtype);return r.typedArrayFromHeap(i).fill(0),i}const fh={kernelName:eu,backendName:"wasm",kernelFunc:hh};/**
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
 */const mh=[Ju,Zu,el,rl,pl,fl,yl,vl,Sl,Fl,El,Nl,Pl,Dl,jl,Hl,Ul,zl,ql,Yl,Zl,nc,sc,ic,uc,lc,cc,pc,fc,mc,gc,_c,kc,xc,Tc,Ec,Cc,Pc,al,Bc,Vc,Hc,Uc,$c,Gc,Kc,Yc,Zc,ep,rp,ip,lp,cp,hp,fp,mp,bp,wp,Sp,Mp,Rp,Ep,Cp,Zs,Dp,jp,Hp,zp,$p,Gp,qp,wl,Yp,Zp,nd,sd,id,od,cd,hd,gd,yd,Tl,vd,kd,xd,Td,Fd,Nd,Od,Pd,Dd,Ld,Vd,Ud,$d,qd,Xd,Yd,Zd,eh,th,ah,oh,ch,ul,dh,fh];for(const a of mh)tu(a);/**
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
 */const vr=Hn();vr.registerFlag("WASM_HAS_SIMD_SUPPORT",async()=>{try{return WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,10,9,1,7,0,65,0,253,15,26,11]))}catch{return!1}});vr.registerFlag("WASM_HAS_MULTITHREAD_SUPPORT",async()=>{if(vr.get("IS_NODE"))return!1;try{return new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)),WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,5,4,1,3,1,1,10,11,1,9,0,65,0,254,16,2,0,26,11]))}catch{return!1}});function wi(a){throw new Error('Could not dynamically require "'+a+'". Please configure the dynamicRequireTargets or/and ignoreDynamicRequires option of @rollup/plugin-commonjs appropriately for this require call to work.')}var zn={},gh={get exports(){return zn},set exports(a){zn=a}};(function(a,e){var r=(()=>{var i=typeof document<"u"&&document.currentScript?document.currentScript.src:void 0;return typeof __filename<"u"&&(i=i||__filename),function(s){s=s||{};function t(){return me.buffer!=Fe&&Ne(me.buffer),He}function u(){return me.buffer!=Fe&&Ne(me.buffer),ut}function p(){return me.buffer!=Fe&&Ne(me.buffer),$t}function h(){return me.buffer!=Fe&&Ne(me.buffer),bt}function d(){return me.buffer!=Fe&&Ne(me.buffer),_t}var n=typeof s<"u"?s:{},f,_;n.ready=new Promise(function(g,x){f=g,_=x});var m;typeof process<"u"&&process.listeners&&(m={uncaughtException:process.listeners("uncaughtException"),unhandledRejection:process.listeners("unhandledRejection")});var y=Object.assign({},n),b=(g,x)=>{throw x},k=typeof window=="object",S=typeof importScripts=="function",I=typeof process=="object"&&typeof process.versions=="object"&&typeof process.versions.node=="string",w=n.ENVIRONMENT_IS_PTHREAD||!1,A="";function O(g){return n.locateFile?n.locateFile(g,A):A+g}var F,P,V;function D(g){if(g instanceof pe)return;Q("exiting due to exception: "+g)}if(I){S?A=Et.dirname(A)+"/":A=__dirname+"/";var j,G;typeof wi=="function"&&(j=Et,G=Et),F=(x,N)=>(x=G.normalize(x),j.readFileSync(x,N?void 0:"utf8")),V=x=>{var N=F(x,!0);return N.buffer||(N=new Uint8Array(N)),N},P=(x,N,H)=>{x=G.normalize(x),j.readFile(x,function(q,ye){q?H(q):N(ye.buffer)})},process.argv.length>1&&process.argv[1].replace(/\\/g,"/"),process.argv.slice(2),process.on("uncaughtException",function(x){if(!(x instanceof pe))throw x}),process.on("unhandledRejection",function(x){throw x}),b=(x,N)=>{if(lt())throw process.exitCode=x,N;D(N),process.exit(x)},n.inspect=function(){return"[Emscripten Module object]"};let g;try{g=Et}catch(x){throw console.error('The "worker_threads" module is not supported in this node.js build - perhaps a newer version is needed?'),x}An.Worker=g.Worker}else(k||S)&&(S?A=self.location.href:typeof document<"u"&&document.currentScript&&(A=document.currentScript.src),typeof i<"u"&&i&&(A=i),A.indexOf("blob:")!==0?A=A.substr(0,A.replace(/[?#].*/,"").lastIndexOf("/")+1):A="",I||(F=g=>{var x=new XMLHttpRequest;return x.open("GET",g,!1),x.send(null),x.responseText},S&&(V=g=>{var x=new XMLHttpRequest;return x.open("GET",g,!1),x.responseType="arraybuffer",x.send(null),new Uint8Array(x.response)}),P=(g,x,N)=>{var H=new XMLHttpRequest;H.open("GET",g,!0),H.responseType="arraybuffer",H.onload=()=>{if(H.status==200||H.status==0&&H.response){x(H.response);return}N()},H.onerror=N,H.send(null)}));I&&typeof performance>"u"&&(An.performance=Et.performance);var X=console.log.bind(console),$=console.warn.bind(console);I&&(X=g=>j.writeSync(1,g+`
`),$=g=>j.writeSync(2,g+`
`));var le=n.print||X,Q=n.printErr||$;Object.assign(n,y),y=null,n.arguments&&n.arguments,n.thisProgram&&n.thisProgram,n.quit&&(b=n.quit);var ce;n.wasmBinary&&(ce=n.wasmBinary);var ke=n.noExitRuntime||!0;typeof WebAssembly!="object"&&At("no native wasm support detected");var me,We,qe=!1,Le;function ft(g,x){g||At(x)}var Be=typeof TextDecoder<"u"?new TextDecoder("utf8"):void 0;function Ct(g,x,N){for(var H=x+N,q=x;g[q]&&!(q>=H);)++q;if(q-x>16&&g.buffer&&Be)return Be.decode(g.buffer instanceof SharedArrayBuffer?g.slice(x,q):g.subarray(x,q));for(var ye="";x<q;){var re=g[x++];if(!(re&128)){ye+=String.fromCharCode(re);continue}var ae=g[x++]&63;if((re&224)==192){ye+=String.fromCharCode((re&31)<<6|ae);continue}var De=g[x++]&63;if((re&240)==224?re=(re&15)<<12|ae<<6|De:re=(re&7)<<18|ae<<12|De<<6|g[x++]&63,re<65536)ye+=String.fromCharCode(re);else{var pt=re-65536;ye+=String.fromCharCode(55296|pt>>10,56320|pt&1023)}}return ye}function xt(g,x){return g?Ct(u(),g,x):""}function zt(g,x,N,H){if(!(H>0))return 0;for(var q=N,ye=N+H-1,re=0;re<g.length;++re){var ae=g.charCodeAt(re);if(ae>=55296&&ae<=57343){var De=g.charCodeAt(++re);ae=65536+((ae&1023)<<10)|De&1023}if(ae<=127){if(N>=ye)break;x[N++]=ae}else if(ae<=2047){if(N+1>=ye)break;x[N++]=192|ae>>6,x[N++]=128|ae&63}else if(ae<=65535){if(N+2>=ye)break;x[N++]=224|ae>>12,x[N++]=128|ae>>6&63,x[N++]=128|ae&63}else{if(N+3>=ye)break;x[N++]=240|ae>>18,x[N++]=128|ae>>12&63,x[N++]=128|ae>>6&63,x[N++]=128|ae&63}}return x[N]=0,N-q}function Ot(g,x,N){return zt(g,u(),x,N)}var Fe,He,ut,$t,bt,_t;w&&(Fe=n.buffer);function Ne(g){Fe=g,n.HEAP8=He=new Int8Array(g),n.HEAP16=new Int16Array(g),n.HEAP32=$t=new Int32Array(g),n.HEAPU8=ut=new Uint8Array(g),n.HEAPU16=new Uint16Array(g),n.HEAPU32=bt=new Uint32Array(g),n.HEAPF32=new Float32Array(g),n.HEAPF64=_t=new Float64Array(g)}var Pt=n.INITIAL_MEMORY||16777216;if(w)me=n.wasmMemory,Fe=n.buffer;else if(n.wasmMemory)me=n.wasmMemory;else if(me=new WebAssembly.Memory({initial:Pt/65536,maximum:32768,shared:!0}),!(me.buffer instanceof SharedArrayBuffer))throw Q("requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag"),I&&console.log("(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and also use a recent version)"),Error("bad memory");me&&(Fe=me.buffer),Pt=Fe.byteLength,Ne(Fe);var Ce,nt=[],Mt=[],mt=[];function lt(){return ke}function Dt(){if(n.preRun)for(typeof n.preRun=="function"&&(n.preRun=[n.preRun]);n.preRun.length;)gt(n.preRun.shift());Ie(nt)}function Gt(){w||Ie(Mt)}function qt(){if(!w){if(n.postRun)for(typeof n.postRun=="function"&&(n.postRun=[n.postRun]);n.postRun.length;)Qe(n.postRun.shift());Ie(mt)}}function gt(g){nt.unshift(g)}function Kt(g){Mt.unshift(g)}function Qe(g){mt.unshift(g)}var dt=0,vt=null;function cn(g){dt++,n.monitorRunDependencies&&n.monitorRunDependencies(dt)}function vn(g){if(dt--,n.monitorRunDependencies&&n.monitorRunDependencies(dt),dt==0&&vt){var x=vt;vt=null,x()}}function At(g){w?postMessage({cmd:"onAbort",arg:g}):n.onAbort&&n.onAbort(g),g="Aborted("+g+")",Q(g),qe=!0,Le=1,g+=". Build with -sASSERTIONS for more info.";var x=new WebAssembly.RuntimeError(g);throw _(x),x}var Tt="data:application/octet-stream;base64,";function Xt(g){return g.startsWith(Tt)}function Ue(g){return g.startsWith("file://")}var Oe;Oe="tfjs-backend-wasm-threaded-simd.wasm",Xt(Oe)||(Oe=O(Oe));function Yt(g){try{if(g==Oe&&ce)return new Uint8Array(ce);if(V)return V(g);throw"both async and sync fetching of the wasm failed"}catch(x){At(x)}}function Me(){if(!ce&&(k||S)){if(typeof fetch=="function"&&!Ue(Oe))return fetch(Oe,{credentials:"same-origin"}).then(function(g){if(!g.ok)throw"failed to load wasm binary file at '"+Oe+"'";return g.arrayBuffer()}).catch(function(){return Yt(Oe)});if(P)return new Promise(function(g,x){P(Oe,function(N){g(new Uint8Array(N))},x)})}return Promise.resolve().then(function(){return Yt(Oe)})}function Qt(){var g={env:ve,wasi_snapshot_preview1:ve};function x(re,ae){var De=re.exports;if(n.asm=De,Lt(n.asm._emscripten_tls_init),Ce=n.asm.__indirect_function_table,Kt(n.asm.__wasm_call_ctors),We=ae,!w){var pt=W.unusedWorkers.length;W.unusedWorkers.forEach(function(Ft){W.loadWasmModuleToWorker(Ft,function(){--pt||vn()})})}}w||cn();function N(re){x(re.instance,re.module)}function H(re){return Me().then(function(ae){return WebAssembly.instantiate(ae,g)}).then(function(ae){return ae}).then(re,function(ae){Q("failed to asynchronously prepare wasm: "+ae),At(ae)})}function q(){return!ce&&typeof WebAssembly.instantiateStreaming=="function"&&!Xt(Oe)&&!Ue(Oe)&&!I&&typeof fetch=="function"?fetch(Oe,{credentials:"same-origin"}).then(function(re){var ae=WebAssembly.instantiateStreaming(re,g);return ae.then(N,function(De){return Q("wasm streaming compile failed: "+De),Q("falling back to ArrayBuffer instantiation"),H(N)})}):H(N)}if(n.instantiateWasm)try{var ye=n.instantiateWasm(g,x);return ye}catch(re){Q("Module.instantiateWasm callback failed with error: "+re),_(re)}return q().catch(_),{}}var Je={};function pe(g){this.name="ExitStatus",this.message="Program terminated with exit("+g+")",this.status=g}function wt(g){var x=W.pthreads[g];delete W.pthreads[g],x.terminate(),ie(g),W.runningWorkers.splice(W.runningWorkers.indexOf(x),1),x.pthread_ptr=0}function Jt(g){var x=W.pthreads[g];x.postMessage({cmd:"cancel"})}function rt(g){var x=W.pthreads[g];ft(x),W.returnWorkerToPool(x)}function kt(g){var x=W.getNewWorker();if(!x)return 6;W.runningWorkers.push(x),W.pthreads[g.pthread_ptr]=x,x.pthread_ptr=g.pthread_ptr;var N={cmd:"run",start_routine:g.startRoutine,arg:g.arg,pthread_ptr:g.pthread_ptr};return x.runPthread=()=>{N.time=performance.now(),x.postMessage(N,g.transferList)},x.loaded&&(x.runPthread(),delete x.runPthread),0}function Zt(g){if(w)return jt(1,1,g);Le=g,lt()||(W.terminateAllThreads(),n.onExit&&n.onExit(g),qe=!0),b(g,new pe(g))}function R(g,x){if(Le=g,!x&&w)throw J(g),"unwind";Zt(g)}var L=R;function U(g){if(g instanceof pe||g=="unwind")return Le;b(1,g)}var W={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init:function(){w?W.initWorker():W.initMainThread()},initMainThread:function(){for(var g=8;g--;)W.allocateUnusedWorker()},initWorker:function(){ke=!1},setExitStatus:function(g){Le=g},terminateAllThreads:function(){for(var g of Object.values(W.pthreads))W.returnWorkerToPool(g);for(var g of W.unusedWorkers)g.terminate();W.unusedWorkers=[]},returnWorkerToPool:function(g){var x=g.pthread_ptr;delete W.pthreads[x],W.unusedWorkers.push(g),W.runningWorkers.splice(W.runningWorkers.indexOf(g),1),g.pthread_ptr=0,ie(x)},receiveObjectTransfer:function(g){},threadInitTLS:function(){W.tlsInitFunctions.forEach(g=>g())},loadWasmModuleToWorker:function(g,x){g.onmessage=N=>{var H=N.data,q=H.cmd;if(g.pthread_ptr&&(W.currentProxiedOperationCallerThread=g.pthread_ptr),H.targetThread&&H.targetThread!=se()){var ye=W.pthreads[H.targetThread];ye?ye.postMessage(H,H.transferList):Q('Internal error! Worker sent a message "'+q+'" to target pthread '+H.targetThread+", but that thread no longer exists!"),W.currentProxiedOperationCallerThread=void 0;return}q==="processProxyingQueue"?kn(H.queue):q==="spawnThread"?kt(H):q==="cleanupThread"?rt(H.thread):q==="killThread"?wt(H.thread):q==="cancelThread"?Jt(H.thread):q==="loaded"?(g.loaded=!0,x&&x(g),g.runPthread&&(g.runPthread(),delete g.runPthread)):q==="print"?le("Thread "+H.threadId+": "+H.text):q==="printErr"?Q("Thread "+H.threadId+": "+H.text):q==="alert"?alert("Thread "+H.threadId+": "+H.text):H.target==="setimmediate"?g.postMessage(H):q==="onAbort"?n.onAbort&&n.onAbort(H.arg):q&&Q("worker sent an unknown command "+q),W.currentProxiedOperationCallerThread=void 0},g.onerror=N=>{var H="worker sent an error!";throw Q(H+" "+N.filename+":"+N.lineno+": "+N.message),N},I&&(g.on("message",function(N){g.onmessage({data:N})}),g.on("error",function(N){g.onerror(N)}),g.on("detachedExit",function(){})),g.postMessage({cmd:"load",urlOrBlob:n.mainScriptUrlOrBlob||i,wasmMemory:me,wasmModule:We})},allocateUnusedWorker:function(){var g=O("tfjs-backend-wasm-threaded-simd.worker.js");W.unusedWorkers.push(new Worker(g))},getNewWorker:function(){return W.unusedWorkers.length==0&&(W.allocateUnusedWorker(),W.loadWasmModuleToWorker(W.unusedWorkers[0])),W.unusedWorkers.pop()}};n.PThread=W;function Ie(g){for(;g.length>0;)g.shift()(n)}function Se(g){var x=we(),N=g();return je(x),N}function ue(){var g=se(),x=p()[g+44>>2],N=p()[g+48>>2],H=x-N;Pe(x,H),je(x)}n.establishStackSpace=ue;function J(g){if(w)return jt(2,0,g);try{L(g)}catch(x){U(x)}}var Ae=[];function Ke(g){var x=Ae[g];return x||(g>=Ae.length&&(Ae.length=g+1),Ae[g]=x=Ce.get(g)),x}function yt(g,x){var N=Ke(g)(x);lt()?W.setExitStatus(N):de(N)}n.invokeEntryPoint=yt;function Lt(g){W.tlsInitFunctions.push(g)}function Bt(g,x){t().set(g,x)}function wn(g){te(g,!S,1,!k),W.threadInitTLS()}function ze(g){w?postMessage({cmd:"cleanupThread",thread:g}):rt(g)}function ct(g,x,N,H){return w?jt(3,1,g,x,N,H):Rt(g,x,N,H)}function Rt(g,x,N,H){if(typeof SharedArrayBuffer>"u")return Q("Current environment does not support SharedArrayBuffer, pthreads are not available!"),6;var q=[],ye=0;if(w&&(q.length===0||ye))return ct(g,x,N,H);var re={startRoutine:N,pthread_ptr:g,arg:H,transferList:q};return w?(re.cmd="spawnThread",postMessage(re,q),0):kt(re)}function Pn(){return 2097152}var Kn=!0;function Xn(){return Kn}function kn(g){Atomics.store(p(),g>>2,1),se()&&Y(g),Atomics.compareExchange(p(),g>>2,1,0)}n.executeNotifiedProxyingQueue=kn;function Yn(g,x,N,H){if(g==x)setTimeout(()=>kn(H));else if(w)postMessage({targetThread:g,cmd:"processProxyingQueue",queue:H});else{var q=W.pthreads[g];if(!q)return;q.postMessage({cmd:"processProxyingQueue",queue:H})}return 1}function Dn(g,x,N){return-1}function Qn(){At("")}function en(g){en.shown||(en.shown={}),en.shown[g]||(en.shown[g]=1,I&&(g="warning: "+g),Q(g))}function In(){I||S||en("Blocking on the main thread is very dangerous, see https://emscripten.org/docs/porting/pthreads.html#blocking-on-the-main-browser-thread")}function Jn(){return Date.now()}function Ln(){return 2147483648}function Zn(){return Ln()}var pn;I?pn=()=>{var g=process.hrtime();return g[0]*1e3+g[1]/1e6}:w?pn=()=>performance.now()-n.__performance_now_clock_drift:pn=()=>performance.now();function dn(g,x,N){u().copyWithin(g,x,x+N)}function er(){return I?Et.cpus().length:navigator.hardwareConcurrency}function jt(g,x){var N=arguments.length-2,H=arguments;return Se(()=>{for(var q=N,ye=Te(q*8),re=ye>>3,ae=0;ae<N;ae++){var De=H[2+ae];d()[re+ae]=De}return K(g,q,ye,x)})}var Sn=[];function Bn(g,x,N){Sn.length=x;for(var H=N>>3,q=0;q<x;q++)Sn[q]=d()[H+q];var ye=g<0,re=ye?Je[-g-1]:Z[g];return re.apply(null,Sn)}function jn(g){try{return me.grow(g-Fe.byteLength+65535>>>16),Ne(me.buffer),1}catch{}}function o(g){var x=u().length;if(g=g>>>0,g<=x)return!1;var N=Ln();if(g>N)return!1;let H=(De,pt)=>De+(pt-De%pt)%pt;for(var q=1;q<=4;q*=2){var ye=x*(1+.2/q);ye=Math.min(ye,g+100663296);var re=Math.min(N,H(Math.max(g,ye),65536)),ae=jn(re);if(ae)return!0}return!1}function l(){throw"unwind"}function c(g){return w?jt(4,1,g):52}function v(g,x,N,H,q){return w?jt(5,1,g,x,N,H,q):70}var M=[null,[],[]];function C(g,x){var N=M[g];x===0||x===10?((g===1?le:Q)(Ct(N,0)),N.length=0):N.push(x)}function T(g,x,N,H){if(w)return jt(6,1,g,x,N,H);for(var q=0,ye=0;ye<N;ye++){var re=h()[x>>2],ae=h()[x+4>>2];x+=8;for(var De=0;De<ae;De++)C(g,u()[re+De]);q+=ae}return h()[H>>2]=q,0}function E(g){var x=n["_"+g];return x}function B(g,x,N,H,q){var ye={string:at=>{var hn=0;if(at!=null&&at!==0){var Nr=(at.length<<2)+1;hn=Te(Nr),Ot(at,hn,Nr)}return hn},array:at=>{var hn=Te(at.length);return Bt(at,hn),hn}};function re(at){return x==="string"?xt(at):x==="boolean"?Boolean(at):at}var ae=E(g),De=[],pt=0;if(H)for(var Ft=0;Ft<H.length;Ft++){var Er=ye[N[Ft]];Er?(pt===0&&(pt=we()),De[Ft]=Er(H[Ft])):De[Ft]=H[Ft]}var tr=ae.apply(null,De);function Hi(at){return pt!==0&&je(pt),re(at)}return tr=Hi(tr),tr}function z(g,x,N,H){N=N||[];var q=N.every(re=>re==="number"||re==="boolean"),ye=x!=="string";return ye&&q&&!H?E(g):function(){return B(g,x,N,arguments)}}W.init();var Z=[null,Zt,J,ct,c,v,T],ve={__emscripten_init_main_thread_js:wn,__emscripten_thread_cleanup:ze,__pthread_create_js:Rt,_emscripten_default_pthread_stack_size:Pn,_emscripten_get_now_is_monotonic:Xn,_emscripten_notify_task_queue:Yn,_emscripten_set_offscreencanvas_size:Dn,abort:Qn,emscripten_check_blocking_allowed:In,emscripten_date_now:Jn,emscripten_get_heap_max:Zn,emscripten_get_now:pn,emscripten_memcpy_big:dn,emscripten_num_logical_cores:er,emscripten_receive_on_main_thread_js:Bn,emscripten_resize_heap:o,emscripten_unwind_to_js_event_loop:l,exit:L,fd_close:c,fd_seek:v,fd_write:T,memory:me||n.wasmMemory};Qt(),n.___wasm_call_ctors=function(){return(n.___wasm_call_ctors=n.asm.__wasm_call_ctors).apply(null,arguments)},n._init=function(){return(n._init=n.asm.init).apply(null,arguments)},n._init_with_threads_count=function(){return(n._init_with_threads_count=n.asm.init_with_threads_count).apply(null,arguments)},n._get_threads_count=function(){return(n._get_threads_count=n.asm.get_threads_count).apply(null,arguments)},n._register_tensor=function(){return(n._register_tensor=n.asm.register_tensor).apply(null,arguments)},n._dispose_data=function(){return(n._dispose_data=n.asm.dispose_data).apply(null,arguments)},n._dispose=function(){return(n._dispose=n.asm.dispose).apply(null,arguments)},n._Abs=function(){return(n._Abs=n.asm.Abs).apply(null,arguments)},n._Add=function(){return(n._Add=n.asm.Add).apply(null,arguments)},n._AddN=function(){return(n._AddN=n.asm.AddN).apply(null,arguments)},n._All=function(){return(n._All=n.asm.All).apply(null,arguments)},n._Any=function(){return(n._Any=n.asm.Any).apply(null,arguments)},n._ArgMax=function(){return(n._ArgMax=n.asm.ArgMax).apply(null,arguments)},n._AvgPool=function(){return(n._AvgPool=n.asm.AvgPool).apply(null,arguments)},n._BatchMatMul=function(){return(n._BatchMatMul=n.asm.BatchMatMul).apply(null,arguments)},n._Ceil=function(){return(n._Ceil=n.asm.Ceil).apply(null,arguments)},n._ClipByValue=function(){return(n._ClipByValue=n.asm.ClipByValue).apply(null,arguments)},n._Conv2D=function(){return(n._Conv2D=n.asm.Conv2D).apply(null,arguments)},n._Conv2DBackpropInput=function(){return(n._Conv2DBackpropInput=n.asm.Conv2DBackpropInput).apply(null,arguments)},n._Cos=function(){return(n._Cos=n.asm.Cos).apply(null,arguments)},n._Cosh=function(){return(n._Cosh=n.asm.Cosh).apply(null,arguments)},n._CropAndResize=function(){return(n._CropAndResize=n.asm.CropAndResize).apply(null,arguments)},n._Cumprod=function(){return(n._Cumprod=n.asm.Cumprod).apply(null,arguments)},n._Cumsum=function(){return(n._Cumsum=n.asm.Cumsum).apply(null,arguments)},n._DepthToSpace=function(){return(n._DepthToSpace=n.asm.DepthToSpace).apply(null,arguments)},n._DepthwiseConv2dNative=function(){return(n._DepthwiseConv2dNative=n.asm.DepthwiseConv2dNative).apply(null,arguments)},n._Elu=function(){return(n._Elu=n.asm.Elu).apply(null,arguments)},n._Equal=function(){return(n._Equal=n.asm.Equal).apply(null,arguments)},n._Exp=function(){return(n._Exp=n.asm.Exp).apply(null,arguments)},n._FlipLeftRight=function(){return(n._FlipLeftRight=n.asm.FlipLeftRight).apply(null,arguments)},n._Floor=function(){return(n._Floor=n.asm.Floor).apply(null,arguments)},n._FloorDiv=function(){return(n._FloorDiv=n.asm.FloorDiv).apply(null,arguments)},n._FusedBatchNorm=function(){return(n._FusedBatchNorm=n.asm.FusedBatchNorm).apply(null,arguments)},n._FusedConv2D=function(){return(n._FusedConv2D=n.asm.FusedConv2D).apply(null,arguments)},n._FusedDepthwiseConv2D=function(){return(n._FusedDepthwiseConv2D=n.asm.FusedDepthwiseConv2D).apply(null,arguments)},n._Gather=function(){return(n._Gather=n.asm.Gather).apply(null,arguments)},n._GatherNd=function(){return(n._GatherNd=n.asm.GatherNd).apply(null,arguments)},n._Greater=function(){return(n._Greater=n.asm.Greater).apply(null,arguments)},n._GreaterEqual=function(){return(n._GreaterEqual=n.asm.GreaterEqual).apply(null,arguments)},n._LeakyRelu=function(){return(n._LeakyRelu=n.asm.LeakyRelu).apply(null,arguments)},n._Less=function(){return(n._Less=n.asm.Less).apply(null,arguments)},n._LessEqual=function(){return(n._LessEqual=n.asm.LessEqual).apply(null,arguments)},n._Log=function(){return(n._Log=n.asm.Log).apply(null,arguments)},n._LogicalAnd=function(){return(n._LogicalAnd=n.asm.LogicalAnd).apply(null,arguments)},n._LogicalNot=function(){return(n._LogicalNot=n.asm.LogicalNot).apply(null,arguments)},n._LogicalOr=function(){return(n._LogicalOr=n.asm.LogicalOr).apply(null,arguments)},n._LogicalXor=function(){return(n._LogicalXor=n.asm.LogicalXor).apply(null,arguments)},n._Max=function(){return(n._Max=n.asm.Max).apply(null,arguments)},n._MaxPool=function(){return(n._MaxPool=n.asm.MaxPool).apply(null,arguments)},n._Maximum=function(){return(n._Maximum=n.asm.Maximum).apply(null,arguments)},n._Mean=function(){return(n._Mean=n.asm.Mean).apply(null,arguments)},n._Min=function(){return(n._Min=n.asm.Min).apply(null,arguments)},n._Minimum=function(){return(n._Minimum=n.asm.Minimum).apply(null,arguments)},n._MirrorPad=function(){return(n._MirrorPad=n.asm.MirrorPad).apply(null,arguments)},n._Multiply=function(){return(n._Multiply=n.asm.Multiply).apply(null,arguments)},n._Neg=function(){return(n._Neg=n.asm.Neg).apply(null,arguments)},n._NonMaxSuppressionV3=function(){return(n._NonMaxSuppressionV3=n.asm.NonMaxSuppressionV3).apply(null,arguments)},n._NonMaxSuppressionV4=function(){return(n._NonMaxSuppressionV4=n.asm.NonMaxSuppressionV4).apply(null,arguments)},n._NonMaxSuppressionV5=function(){return(n._NonMaxSuppressionV5=n.asm.NonMaxSuppressionV5).apply(null,arguments)},n._NotEqual=function(){return(n._NotEqual=n.asm.NotEqual).apply(null,arguments)},n._OneHot=function(){return(n._OneHot=n.asm.OneHot).apply(null,arguments)},n._PadV2=function(){return(n._PadV2=n.asm.PadV2).apply(null,arguments)},n._Pow=function(){return(n._Pow=n.asm.Pow).apply(null,arguments)},n._Prelu=function(){return(n._Prelu=n.asm.Prelu).apply(null,arguments)},n._Prod=function(){return(n._Prod=n.asm.Prod).apply(null,arguments)},n._RealDiv=function(){return(n._RealDiv=n.asm.RealDiv).apply(null,arguments)},n._Relu=function(){return(n._Relu=n.asm.Relu).apply(null,arguments)},n._Relu6=function(){return(n._Relu6=n.asm.Relu6).apply(null,arguments)},n._ResizeBilinear=function(){return(n._ResizeBilinear=n.asm.ResizeBilinear).apply(null,arguments)},n._ResizeNearestNeighbor=function(){return(n._ResizeNearestNeighbor=n.asm.ResizeNearestNeighbor).apply(null,arguments)},n._Reverse=function(){return(n._Reverse=n.asm.Reverse).apply(null,arguments)},n._RotateWithOffset=function(){return(n._RotateWithOffset=n.asm.RotateWithOffset).apply(null,arguments)},n._Round=function(){return(n._Round=n.asm.Round).apply(null,arguments)},n._Rsqrt=function(){return(n._Rsqrt=n.asm.Rsqrt).apply(null,arguments)},n._ScatterNd=function(){return(n._ScatterNd=n.asm.ScatterNd).apply(null,arguments)},n._SelectV2=function(){return(n._SelectV2=n.asm.SelectV2).apply(null,arguments)},n._Sigmoid=function(){return(n._Sigmoid=n.asm.Sigmoid).apply(null,arguments)},n._Sin=function(){return(n._Sin=n.asm.Sin).apply(null,arguments)},n._Softmax=function(){return(n._Softmax=n.asm.Softmax).apply(null,arguments)},n._SparseFillEmptyRows=function(){return(n._SparseFillEmptyRows=n.asm.SparseFillEmptyRows).apply(null,arguments)},n._SparseReshape=function(){return(n._SparseReshape=n.asm.SparseReshape).apply(null,arguments)},n._SparseSegmentReduction=function(){return(n._SparseSegmentReduction=n.asm.SparseSegmentReduction).apply(null,arguments)},n._Sqrt=function(){return(n._Sqrt=n.asm.Sqrt).apply(null,arguments)},n._Square=function(){return(n._Square=n.asm.Square).apply(null,arguments)},n._SquaredDifference=function(){return(n._SquaredDifference=n.asm.SquaredDifference).apply(null,arguments)},n._Step=function(){return(n._Step=n.asm.Step).apply(null,arguments)},n._StridedSlice=function(){return(n._StridedSlice=n.asm.StridedSlice).apply(null,arguments)},n._Sub=function(){return(n._Sub=n.asm.Sub).apply(null,arguments)},n._Sum=function(){return(n._Sum=n.asm.Sum).apply(null,arguments)},n._Tan=function(){return(n._Tan=n.asm.Tan).apply(null,arguments)},n._Tanh=function(){return(n._Tanh=n.asm.Tanh).apply(null,arguments)},n._Tile=function(){return(n._Tile=n.asm.Tile).apply(null,arguments)},n._TopK=function(){return(n._TopK=n.asm.TopK).apply(null,arguments)},n._Transform=function(){return(n._Transform=n.asm.Transform).apply(null,arguments)},n._Transpose=function(){return(n._Transpose=n.asm.Transpose).apply(null,arguments)},n.__FusedMatMul=function(){return(n.__FusedMatMul=n.asm._FusedMatMul).apply(null,arguments)},n._malloc=function(){return(n._malloc=n.asm.malloc).apply(null,arguments)},n._free=function(){return(n._free=n.asm.free).apply(null,arguments)},n.__emscripten_tls_init=function(){return(n.__emscripten_tls_init=n.asm._emscripten_tls_init).apply(null,arguments)};var se=n._pthread_self=function(){return(se=n._pthread_self=n.asm.pthread_self).apply(null,arguments)};n.___errno_location=function(){return(n.___errno_location=n.asm.__errno_location).apply(null,arguments)};var te=n.__emscripten_thread_init=function(){return(te=n.__emscripten_thread_init=n.asm._emscripten_thread_init).apply(null,arguments)};n.__emscripten_thread_crashed=function(){return(n.__emscripten_thread_crashed=n.asm._emscripten_thread_crashed).apply(null,arguments)},n._emscripten_main_thread_process_queued_calls=function(){return(n._emscripten_main_thread_process_queued_calls=n.asm.emscripten_main_thread_process_queued_calls).apply(null,arguments)},n._emscripten_main_browser_thread_id=function(){return(n._emscripten_main_browser_thread_id=n.asm.emscripten_main_browser_thread_id).apply(null,arguments)};var K=n._emscripten_run_in_main_runtime_thread_js=function(){return(K=n._emscripten_run_in_main_runtime_thread_js=n.asm.emscripten_run_in_main_runtime_thread_js).apply(null,arguments)};n._emscripten_dispatch_to_thread_=function(){return(n._emscripten_dispatch_to_thread_=n.asm.emscripten_dispatch_to_thread_).apply(null,arguments)};var Y=n.__emscripten_proxy_execute_task_queue=function(){return(Y=n.__emscripten_proxy_execute_task_queue=n.asm._emscripten_proxy_execute_task_queue).apply(null,arguments)},ie=n.__emscripten_thread_free_data=function(){return(ie=n.__emscripten_thread_free_data=n.asm._emscripten_thread_free_data).apply(null,arguments)},de=n.__emscripten_thread_exit=function(){return(de=n.__emscripten_thread_exit=n.asm._emscripten_thread_exit).apply(null,arguments)},Pe=n._emscripten_stack_set_limits=function(){return(Pe=n._emscripten_stack_set_limits=n.asm.emscripten_stack_set_limits).apply(null,arguments)},we=n.stackSave=function(){return(we=n.stackSave=n.asm.stackSave).apply(null,arguments)},je=n.stackRestore=function(){return(je=n.stackRestore=n.asm.stackRestore).apply(null,arguments)},Te=n.stackAlloc=function(){return(Te=n.stackAlloc=n.asm.stackAlloc).apply(null,arguments)};n.dynCall_iijjiiii=function(){return(n.dynCall_iijjiiii=n.asm.dynCall_iijjiiii).apply(null,arguments)},n.dynCall_jiji=function(){return(n.dynCall_jiji=n.asm.dynCall_jiji).apply(null,arguments)},n.keepRuntimeAlive=lt,n.wasmMemory=me,n.cwrap=z,n.ExitStatus=pe,n.PThread=W;var ne;vt=function g(){ne||fe(),ne||(vt=g)};function fe(g){if(dt>0)return;if(w){f(n),Gt(),postMessage({cmd:"loaded"});return}if(Dt(),dt>0)return;function x(){ne||(ne=!0,n.calledRun=!0,!qe&&(Gt(),f(n),n.onRuntimeInitialized&&n.onRuntimeInitialized(),qt()))}n.setStatus?(n.setStatus("Running..."),setTimeout(function(){setTimeout(function(){n.setStatus("")},1),x()},1)):x()}if(n.preInit)for(typeof n.preInit=="function"&&(n.preInit=[n.preInit]);n.preInit.length>0;)n.preInit.pop()();fe();var he;m&&(he={uncaughtException:process.listeners("uncaughtException").filter(function(g){return!m.uncaughtException.indexOf(g)>-1}),unhandledRejection:process.listeners("unhandledRejection").filter(function(g){return!m.unhandledRejection.indexOf(g)>-1})});var oe;if(typeof WasmBackendModule<"u")oe=WasmBackendModule;else if(typeof s<"u")oe=s;else throw new Error("Could not find wasm module in post.js");if(he){var ge=oe._dispose;oe._dispose=function(){ge(),he.uncaughtException.forEach(function(g){process.removeListener("uncaughtException",g)}),he.unhandledRejection.forEach(function(g){process.removeListener("unhandledRejection",g)})}}return s.ready}})();a.exports=r})(gh);const ki=zn,yh=ys({__proto__:null,default:ki},[zn]);var bh=`"use strict";var Module={};var ENVIRONMENT_IS_NODE=typeof process=="object"&&typeof process.versions=="object"&&typeof process.versions.node=="string";if(ENVIRONMENT_IS_NODE){var nodeWorkerThreads=require("worker_threads");var parentPort=nodeWorkerThreads.parentPort;parentPort.on("message",data=>onmessage({data:data}));var fs=require("fs");Object.assign(global,{self:global,require:require,Module:Module,location:{href:__filename},Worker:nodeWorkerThreads.Worker,importScripts:function(f){(0,eval)(fs.readFileSync(f,"utf8"))},postMessage:function(msg){parentPort.postMessage(msg)},performance:global.performance||{now:function(){return Date.now()}}})}var initializedJS=false;var pendingNotifiedProxyingQueues=[];function threadPrintErr(){var text=Array.prototype.slice.call(arguments).join(" ");if(ENVIRONMENT_IS_NODE){fs.writeSync(2,text+"
");return}console.error(text)}function threadAlert(){var text=Array.prototype.slice.call(arguments).join(" ");postMessage({cmd:"alert",text:text,threadId:Module["_pthread_self"]()})}var err=threadPrintErr;self.alert=threadAlert;Module["instantiateWasm"]=(info,receiveInstance)=>{var instance=new WebAssembly.Instance(Module["wasmModule"],info);receiveInstance(instance);Module["wasmModule"]=null;return instance.exports};self.onunhandledrejection=e=>{throw e.reason??e};self.onmessage=e=>{try{if(e.data.cmd==="load"){Module["wasmModule"]=e.data.wasmModule;Module["wasmMemory"]=e.data.wasmMemory;Module["buffer"]=Module["wasmMemory"].buffer;Module["ENVIRONMENT_IS_PTHREAD"]=true;if(typeof e.data.urlOrBlob=="string"){importScripts(e.data.urlOrBlob)}else{var objectUrl=URL.createObjectURL(e.data.urlOrBlob);importScripts(objectUrl);URL.revokeObjectURL(objectUrl)}WasmBackendModuleThreadedSimd(Module).then(function(instance){Module=instance})}else if(e.data.cmd==="run"){Module["__performance_now_clock_drift"]=performance.now()-e.data.time;Module["__emscripten_thread_init"](e.data.pthread_ptr,0,0,1);Module["establishStackSpace"]();Module["PThread"].receiveObjectTransfer(e.data);Module["PThread"].threadInitTLS();if(!initializedJS){pendingNotifiedProxyingQueues.forEach(queue=>{Module["executeNotifiedProxyingQueue"](queue)});pendingNotifiedProxyingQueues=[];initializedJS=true}try{Module["invokeEntryPoint"](e.data.start_routine,e.data.arg)}catch(ex){if(ex!="unwind"){if(ex instanceof Module["ExitStatus"]){if(Module["keepRuntimeAlive"]()){}else{Module["__emscripten_thread_exit"](ex.status)}}else{throw ex}}}}else if(e.data.cmd==="cancel"){if(Module["_pthread_self"]()){Module["__emscripten_thread_exit"](-1)}}else if(e.data.target==="setimmediate"){}else if(e.data.cmd==="processProxyingQueue"){if(initializedJS){Module["executeNotifiedProxyingQueue"](e.data.queue)}else{pendingNotifiedProxyingQueues.push(e.data.queue)}}else if(e.data.cmd){err("worker.js received unknown command "+e.data.cmd);err(e.data)}}catch(ex){if(Module["__emscripten_thread_crashed"]){Module["__emscripten_thread_crashed"]()}throw ex}};`,$n={},_h={get exports(){return $n},set exports(a){$n=a}};(function(a,e){var r=(()=>{var i=typeof document<"u"&&document.currentScript?document.currentScript.src:void 0;return typeof __filename<"u"&&(i=i||__filename),function(s){s=s||{};var t=typeof s<"u"?s:{},u,p;t.ready=new Promise(function(R,L){u=R,p=L});var h;typeof process<"u"&&process.listeners&&(h={uncaughtException:process.listeners("uncaughtException"),unhandledRejection:process.listeners("unhandledRejection")});var d=Object.assign({},t),n=typeof window=="object",f=typeof importScripts=="function",_=typeof process=="object"&&typeof process.versions=="object"&&typeof process.versions.node=="string",m="";function y(R){return t.locateFile?t.locateFile(R,m):m+R}var b,k,S;if(_){f?m=Et.dirname(m)+"/":m=__dirname+"/";var I,w;typeof wi=="function"&&(I=Et,w=Et),b=(R,L)=>(R=w.normalize(R),I.readFileSync(R,L?void 0:"utf8")),S=R=>{var L=b(R,!0);return L.buffer||(L=new Uint8Array(L)),L},k=(R,L,U)=>{R=w.normalize(R),I.readFile(R,function(W,Ie){W?U(W):L(Ie.buffer)})},process.argv.length>1&&process.argv[1].replace(/\\/g,"/"),process.argv.slice(2),process.on("uncaughtException",function(R){if(!(R instanceof lt))throw R}),process.on("unhandledRejection",function(R){throw R}),t.inspect=function(){return"[Emscripten Module object]"}}else(n||f)&&(f?m=self.location.href:typeof document<"u"&&document.currentScript&&(m=document.currentScript.src),i&&(m=i),m.indexOf("blob:")!==0?m=m.substr(0,m.replace(/[?#].*/,"").lastIndexOf("/")+1):m="",b=R=>{var L=new XMLHttpRequest;return L.open("GET",R,!1),L.send(null),L.responseText},f&&(S=R=>{var L=new XMLHttpRequest;return L.open("GET",R,!1),L.responseType="arraybuffer",L.send(null),new Uint8Array(L.response)}),k=(R,L,U)=>{var W=new XMLHttpRequest;W.open("GET",R,!0),W.responseType="arraybuffer",W.onload=()=>{if(W.status==200||W.status==0&&W.response){L(W.response);return}U()},W.onerror=U,W.send(null)});var A=t.print||console.log.bind(console),O=t.printErr||console.warn.bind(console);Object.assign(t,d),d=null,t.arguments&&t.arguments,t.thisProgram&&t.thisProgram,t.quit&&t.quit;var F;t.wasmBinary&&(F=t.wasmBinary),t.noExitRuntime,typeof WebAssembly!="object"&&bt("no native wasm support detected");var P,V=!1,D=typeof TextDecoder<"u"?new TextDecoder("utf8"):void 0;function j(R,L,U){for(var W=L+U,Ie=L;R[Ie]&&!(Ie>=W);)++Ie;if(Ie-L>16&&R.buffer&&D)return D.decode(R.subarray(L,Ie));for(var Se="";L<Ie;){var ue=R[L++];if(!(ue&128)){Se+=String.fromCharCode(ue);continue}var J=R[L++]&63;if((ue&224)==192){Se+=String.fromCharCode((ue&31)<<6|J);continue}var Ae=R[L++]&63;if((ue&240)==224?ue=(ue&15)<<12|J<<6|Ae:ue=(ue&7)<<18|J<<12|Ae<<6|R[L++]&63,ue<65536)Se+=String.fromCharCode(ue);else{var Ke=ue-65536;Se+=String.fromCharCode(55296|Ke>>10,56320|Ke&1023)}}return Se}function G(R,L){return R?j(ce,R,L):""}function X(R,L,U,W){if(!(W>0))return 0;for(var Ie=U,Se=U+W-1,ue=0;ue<R.length;++ue){var J=R.charCodeAt(ue);if(J>=55296&&J<=57343){var Ae=R.charCodeAt(++ue);J=65536+((J&1023)<<10)|Ae&1023}if(J<=127){if(U>=Se)break;L[U++]=J}else if(J<=2047){if(U+1>=Se)break;L[U++]=192|J>>6,L[U++]=128|J&63}else if(J<=65535){if(U+2>=Se)break;L[U++]=224|J>>12,L[U++]=128|J>>6&63,L[U++]=128|J&63}else{if(U+3>=Se)break;L[U++]=240|J>>18,L[U++]=128|J>>12&63,L[U++]=128|J>>6&63,L[U++]=128|J&63}}return L[U]=0,U-Ie}function $(R,L,U){return X(R,ce,L,U)}var le,Q,ce,ke;function me(R){le=R,t.HEAP8=Q=new Int8Array(R),t.HEAP16=new Int16Array(R),t.HEAP32=new Int32Array(R),t.HEAPU8=ce=new Uint8Array(R),t.HEAPU16=new Uint16Array(R),t.HEAPU32=ke=new Uint32Array(R),t.HEAPF32=new Float32Array(R),t.HEAPF64=new Float64Array(R)}t.INITIAL_MEMORY;var We=[],qe=[],Le=[];function ft(){if(t.preRun)for(typeof t.preRun=="function"&&(t.preRun=[t.preRun]);t.preRun.length;)xt(t.preRun.shift());Dt(We)}function Be(){Dt(qe)}function Ct(){if(t.postRun)for(typeof t.postRun=="function"&&(t.postRun=[t.postRun]);t.postRun.length;)Ot(t.postRun.shift());Dt(Le)}function xt(R){We.unshift(R)}function zt(R){qe.unshift(R)}function Ot(R){Le.unshift(R)}var Fe=0,He=null;function ut(R){Fe++,t.monitorRunDependencies&&t.monitorRunDependencies(Fe)}function $t(R){if(Fe--,t.monitorRunDependencies&&t.monitorRunDependencies(Fe),Fe==0&&He){var L=He;He=null,L()}}function bt(R){t.onAbort&&t.onAbort(R),R="Aborted("+R+")",O(R),V=!0,R+=". Build with -sASSERTIONS for more info.";var L=new WebAssembly.RuntimeError(R);throw p(L),L}var _t="data:application/octet-stream;base64,";function Ne(R){return R.startsWith(_t)}function Pt(R){return R.startsWith("file://")}var Ce;Ce="tfjs-backend-wasm.wasm",Ne(Ce)||(Ce=y(Ce));function nt(R){try{if(R==Ce&&F)return new Uint8Array(F);if(S)return S(R);throw"both async and sync fetching of the wasm failed"}catch(L){bt(L)}}function Mt(){if(!F&&(n||f)){if(typeof fetch=="function"&&!Pt(Ce))return fetch(Ce,{credentials:"same-origin"}).then(function(R){if(!R.ok)throw"failed to load wasm binary file at '"+Ce+"'";return R.arrayBuffer()}).catch(function(){return nt(Ce)});if(k)return new Promise(function(R,L){k(Ce,function(U){R(new Uint8Array(U))},L)})}return Promise.resolve().then(function(){return nt(Ce)})}function mt(){var R={env:Me,wasi_snapshot_preview1:Me};function L(ue,J){var Ae=ue.exports;t.asm=Ae,P=t.asm.memory,me(P.buffer),t.asm.__indirect_function_table,zt(t.asm.__wasm_call_ctors),$t()}ut();function U(ue){L(ue.instance)}function W(ue){return Mt().then(function(J){return WebAssembly.instantiate(J,R)}).then(function(J){return J}).then(ue,function(J){O("failed to asynchronously prepare wasm: "+J),bt(J)})}function Ie(){return!F&&typeof WebAssembly.instantiateStreaming=="function"&&!Ne(Ce)&&!Pt(Ce)&&!_&&typeof fetch=="function"?fetch(Ce,{credentials:"same-origin"}).then(function(ue){var J=WebAssembly.instantiateStreaming(ue,R);return J.then(U,function(Ae){return O("wasm streaming compile failed: "+Ae),O("falling back to ArrayBuffer instantiation"),W(U)})}):W(U)}if(t.instantiateWasm)try{var Se=t.instantiateWasm(R,L);return Se}catch(ue){O("Module.instantiateWasm callback failed with error: "+ue),p(ue)}return Ie().catch(p),{}}function lt(R){this.name="ExitStatus",this.message="Program terminated with exit("+R+")",this.status=R}function Dt(R){for(;R.length>0;)R.shift()(t)}function Gt(R,L){Q.set(R,L)}function qt(){bt("")}function gt(){return 2147483648}function Kt(){return gt()}function Qe(R,L,U){ce.copyWithin(R,L,L+U)}function dt(R){try{return P.grow(R-le.byteLength+65535>>>16),me(P.buffer),1}catch{}}function vt(R){var L=ce.length;R=R>>>0;var U=gt();if(R>U)return!1;let W=(Ae,Ke)=>Ae+(Ke-Ae%Ke)%Ke;for(var Ie=1;Ie<=4;Ie*=2){var Se=L*(1+.2/Ie);Se=Math.min(Se,R+100663296);var ue=Math.min(U,W(Math.max(R,Se),65536)),J=dt(ue);if(J)return!0}return!1}function cn(R){return 52}function vn(R,L,U,W,Ie){return 70}var At=[null,[],[]];function Tt(R,L){var U=At[R];L===0||L===10?((R===1?A:O)(j(U,0)),U.length=0):U.push(L)}function Xt(R,L,U,W){for(var Ie=0,Se=0;Se<U;Se++){var ue=ke[L>>2],J=ke[L+4>>2];L+=8;for(var Ae=0;Ae<J;Ae++)Tt(R,ce[ue+Ae]);Ie+=J}return ke[W>>2]=Ie,0}function Ue(R){var L=t["_"+R];return L}function Oe(R,L,U,W,Ie){var Se={string:ze=>{var ct=0;if(ze!=null&&ze!==0){var Rt=(ze.length<<2)+1;ct=pe(Rt),$(ze,ct,Rt)}return ct},array:ze=>{var ct=pe(ze.length);return Gt(ze,ct),ct}};function ue(ze){return L==="string"?G(ze):L==="boolean"?Boolean(ze):ze}var J=Ue(R),Ae=[],Ke=0;if(W)for(var yt=0;yt<W.length;yt++){var Lt=Se[U[yt]];Lt?(Ke===0&&(Ke=Qt()),Ae[yt]=Lt(W[yt])):Ae[yt]=W[yt]}var Bt=J.apply(null,Ae);function wn(ze){return Ke!==0&&Je(Ke),ue(ze)}return Bt=wn(Bt),Bt}function Yt(R,L,U,W){U=U||[];var Ie=U.every(ue=>ue==="number"||ue==="boolean"),Se=L!=="string";return Se&&Ie&&!W?Ue(R):function(){return Oe(R,L,U,arguments)}}var Me={abort:qt,emscripten_get_heap_max:Kt,emscripten_memcpy_big:Qe,emscripten_resize_heap:vt,fd_close:cn,fd_seek:vn,fd_write:Xt};mt(),t.___wasm_call_ctors=function(){return(t.___wasm_call_ctors=t.asm.__wasm_call_ctors).apply(null,arguments)},t._init=function(){return(t._init=t.asm.init).apply(null,arguments)},t._init_with_threads_count=function(){return(t._init_with_threads_count=t.asm.init_with_threads_count).apply(null,arguments)},t._get_threads_count=function(){return(t._get_threads_count=t.asm.get_threads_count).apply(null,arguments)},t._register_tensor=function(){return(t._register_tensor=t.asm.register_tensor).apply(null,arguments)},t._dispose_data=function(){return(t._dispose_data=t.asm.dispose_data).apply(null,arguments)},t._dispose=function(){return(t._dispose=t.asm.dispose).apply(null,arguments)},t._Abs=function(){return(t._Abs=t.asm.Abs).apply(null,arguments)},t._Add=function(){return(t._Add=t.asm.Add).apply(null,arguments)},t._AddN=function(){return(t._AddN=t.asm.AddN).apply(null,arguments)},t._All=function(){return(t._All=t.asm.All).apply(null,arguments)},t._Any=function(){return(t._Any=t.asm.Any).apply(null,arguments)},t._ArgMax=function(){return(t._ArgMax=t.asm.ArgMax).apply(null,arguments)},t._AvgPool=function(){return(t._AvgPool=t.asm.AvgPool).apply(null,arguments)},t._BatchMatMul=function(){return(t._BatchMatMul=t.asm.BatchMatMul).apply(null,arguments)},t._Ceil=function(){return(t._Ceil=t.asm.Ceil).apply(null,arguments)},t._ClipByValue=function(){return(t._ClipByValue=t.asm.ClipByValue).apply(null,arguments)},t._Conv2D=function(){return(t._Conv2D=t.asm.Conv2D).apply(null,arguments)},t._Conv2DBackpropInput=function(){return(t._Conv2DBackpropInput=t.asm.Conv2DBackpropInput).apply(null,arguments)},t._Cos=function(){return(t._Cos=t.asm.Cos).apply(null,arguments)},t._Cosh=function(){return(t._Cosh=t.asm.Cosh).apply(null,arguments)},t._CropAndResize=function(){return(t._CropAndResize=t.asm.CropAndResize).apply(null,arguments)},t._Cumprod=function(){return(t._Cumprod=t.asm.Cumprod).apply(null,arguments)},t._Cumsum=function(){return(t._Cumsum=t.asm.Cumsum).apply(null,arguments)},t._DepthToSpace=function(){return(t._DepthToSpace=t.asm.DepthToSpace).apply(null,arguments)},t._DepthwiseConv2dNative=function(){return(t._DepthwiseConv2dNative=t.asm.DepthwiseConv2dNative).apply(null,arguments)},t._Elu=function(){return(t._Elu=t.asm.Elu).apply(null,arguments)},t._Equal=function(){return(t._Equal=t.asm.Equal).apply(null,arguments)},t._Exp=function(){return(t._Exp=t.asm.Exp).apply(null,arguments)},t._FlipLeftRight=function(){return(t._FlipLeftRight=t.asm.FlipLeftRight).apply(null,arguments)},t._Floor=function(){return(t._Floor=t.asm.Floor).apply(null,arguments)},t._FloorDiv=function(){return(t._FloorDiv=t.asm.FloorDiv).apply(null,arguments)},t._FusedBatchNorm=function(){return(t._FusedBatchNorm=t.asm.FusedBatchNorm).apply(null,arguments)},t._FusedConv2D=function(){return(t._FusedConv2D=t.asm.FusedConv2D).apply(null,arguments)},t._FusedDepthwiseConv2D=function(){return(t._FusedDepthwiseConv2D=t.asm.FusedDepthwiseConv2D).apply(null,arguments)},t._Gather=function(){return(t._Gather=t.asm.Gather).apply(null,arguments)},t._GatherNd=function(){return(t._GatherNd=t.asm.GatherNd).apply(null,arguments)},t._Greater=function(){return(t._Greater=t.asm.Greater).apply(null,arguments)},t._GreaterEqual=function(){return(t._GreaterEqual=t.asm.GreaterEqual).apply(null,arguments)},t._LeakyRelu=function(){return(t._LeakyRelu=t.asm.LeakyRelu).apply(null,arguments)},t._Less=function(){return(t._Less=t.asm.Less).apply(null,arguments)},t._LessEqual=function(){return(t._LessEqual=t.asm.LessEqual).apply(null,arguments)},t._Log=function(){return(t._Log=t.asm.Log).apply(null,arguments)},t._LogicalAnd=function(){return(t._LogicalAnd=t.asm.LogicalAnd).apply(null,arguments)},t._LogicalNot=function(){return(t._LogicalNot=t.asm.LogicalNot).apply(null,arguments)},t._LogicalOr=function(){return(t._LogicalOr=t.asm.LogicalOr).apply(null,arguments)},t._LogicalXor=function(){return(t._LogicalXor=t.asm.LogicalXor).apply(null,arguments)},t._Max=function(){return(t._Max=t.asm.Max).apply(null,arguments)},t._MaxPool=function(){return(t._MaxPool=t.asm.MaxPool).apply(null,arguments)},t._Maximum=function(){return(t._Maximum=t.asm.Maximum).apply(null,arguments)},t._Mean=function(){return(t._Mean=t.asm.Mean).apply(null,arguments)},t._Min=function(){return(t._Min=t.asm.Min).apply(null,arguments)},t._Minimum=function(){return(t._Minimum=t.asm.Minimum).apply(null,arguments)},t._MirrorPad=function(){return(t._MirrorPad=t.asm.MirrorPad).apply(null,arguments)},t._Multiply=function(){return(t._Multiply=t.asm.Multiply).apply(null,arguments)},t._Neg=function(){return(t._Neg=t.asm.Neg).apply(null,arguments)},t._NonMaxSuppressionV3=function(){return(t._NonMaxSuppressionV3=t.asm.NonMaxSuppressionV3).apply(null,arguments)},t._NonMaxSuppressionV4=function(){return(t._NonMaxSuppressionV4=t.asm.NonMaxSuppressionV4).apply(null,arguments)},t._NonMaxSuppressionV5=function(){return(t._NonMaxSuppressionV5=t.asm.NonMaxSuppressionV5).apply(null,arguments)},t._NotEqual=function(){return(t._NotEqual=t.asm.NotEqual).apply(null,arguments)},t._OneHot=function(){return(t._OneHot=t.asm.OneHot).apply(null,arguments)},t._PadV2=function(){return(t._PadV2=t.asm.PadV2).apply(null,arguments)},t._Pow=function(){return(t._Pow=t.asm.Pow).apply(null,arguments)},t._Prelu=function(){return(t._Prelu=t.asm.Prelu).apply(null,arguments)},t._Prod=function(){return(t._Prod=t.asm.Prod).apply(null,arguments)},t._RealDiv=function(){return(t._RealDiv=t.asm.RealDiv).apply(null,arguments)},t._Relu=function(){return(t._Relu=t.asm.Relu).apply(null,arguments)},t._Relu6=function(){return(t._Relu6=t.asm.Relu6).apply(null,arguments)},t._ResizeBilinear=function(){return(t._ResizeBilinear=t.asm.ResizeBilinear).apply(null,arguments)},t._ResizeNearestNeighbor=function(){return(t._ResizeNearestNeighbor=t.asm.ResizeNearestNeighbor).apply(null,arguments)},t._Reverse=function(){return(t._Reverse=t.asm.Reverse).apply(null,arguments)},t._RotateWithOffset=function(){return(t._RotateWithOffset=t.asm.RotateWithOffset).apply(null,arguments)},t._Round=function(){return(t._Round=t.asm.Round).apply(null,arguments)},t._Rsqrt=function(){return(t._Rsqrt=t.asm.Rsqrt).apply(null,arguments)},t._ScatterNd=function(){return(t._ScatterNd=t.asm.ScatterNd).apply(null,arguments)},t._SelectV2=function(){return(t._SelectV2=t.asm.SelectV2).apply(null,arguments)},t._Sigmoid=function(){return(t._Sigmoid=t.asm.Sigmoid).apply(null,arguments)},t._Sin=function(){return(t._Sin=t.asm.Sin).apply(null,arguments)},t._Softmax=function(){return(t._Softmax=t.asm.Softmax).apply(null,arguments)},t._SparseFillEmptyRows=function(){return(t._SparseFillEmptyRows=t.asm.SparseFillEmptyRows).apply(null,arguments)},t._SparseReshape=function(){return(t._SparseReshape=t.asm.SparseReshape).apply(null,arguments)},t._SparseSegmentReduction=function(){return(t._SparseSegmentReduction=t.asm.SparseSegmentReduction).apply(null,arguments)},t._Sqrt=function(){return(t._Sqrt=t.asm.Sqrt).apply(null,arguments)},t._Square=function(){return(t._Square=t.asm.Square).apply(null,arguments)},t._SquaredDifference=function(){return(t._SquaredDifference=t.asm.SquaredDifference).apply(null,arguments)},t._Step=function(){return(t._Step=t.asm.Step).apply(null,arguments)},t._StridedSlice=function(){return(t._StridedSlice=t.asm.StridedSlice).apply(null,arguments)},t._Sub=function(){return(t._Sub=t.asm.Sub).apply(null,arguments)},t._Sum=function(){return(t._Sum=t.asm.Sum).apply(null,arguments)},t._Tan=function(){return(t._Tan=t.asm.Tan).apply(null,arguments)},t._Tanh=function(){return(t._Tanh=t.asm.Tanh).apply(null,arguments)},t._Tile=function(){return(t._Tile=t.asm.Tile).apply(null,arguments)},t._TopK=function(){return(t._TopK=t.asm.TopK).apply(null,arguments)},t._Transform=function(){return(t._Transform=t.asm.Transform).apply(null,arguments)},t._Transpose=function(){return(t._Transpose=t.asm.Transpose).apply(null,arguments)},t.__FusedMatMul=function(){return(t.__FusedMatMul=t.asm._FusedMatMul).apply(null,arguments)},t._malloc=function(){return(t._malloc=t.asm.malloc).apply(null,arguments)},t._free=function(){return(t._free=t.asm.free).apply(null,arguments)},t.___errno_location=function(){return(t.___errno_location=t.asm.__errno_location).apply(null,arguments)};var Qt=t.stackSave=function(){return(Qt=t.stackSave=t.asm.stackSave).apply(null,arguments)},Je=t.stackRestore=function(){return(Je=t.stackRestore=t.asm.stackRestore).apply(null,arguments)},pe=t.stackAlloc=function(){return(pe=t.stackAlloc=t.asm.stackAlloc).apply(null,arguments)};t.dynCall_iijjiiii=function(){return(t.dynCall_iijjiiii=t.asm.dynCall_iijjiiii).apply(null,arguments)},t.dynCall_jiji=function(){return(t.dynCall_jiji=t.asm.dynCall_jiji).apply(null,arguments)},t.cwrap=Yt;var wt;He=function R(){wt||Jt(),wt||(He=R)};function Jt(R){if(Fe>0||(ft(),Fe>0))return;function L(){wt||(wt=!0,t.calledRun=!0,!V&&(Be(),u(t),t.onRuntimeInitialized&&t.onRuntimeInitialized(),Ct()))}t.setStatus?(t.setStatus("Running..."),setTimeout(function(){setTimeout(function(){t.setStatus("")},1),L()},1)):L()}if(t.preInit)for(typeof t.preInit=="function"&&(t.preInit=[t.preInit]);t.preInit.length>0;)t.preInit.pop()();Jt();var rt;h&&(rt={uncaughtException:process.listeners("uncaughtException").filter(function(R){return!h.uncaughtException.indexOf(R)>-1}),unhandledRejection:process.listeners("unhandledRejection").filter(function(R){return!h.unhandledRejection.indexOf(R)>-1})});var kt;if(typeof s<"u")kt=s;else if(typeof WasmBackendModuleThreadedSimd<"u")kt=WasmBackendModuleThreadedSimd;else throw new Error("Could not find wasm module in post.js");if(rt){var Zt=kt._dispose;kt._dispose=function(){Zt(),rt.uncaughtException.forEach(function(R){process.removeListener("uncaughtException",R)}),rt.unhandledRejection.forEach(function(R){process.removeListener("unhandledRejection",R)})}}return s.ready}})();a.exports=r})(_h);const Ii=$n,vh=ys({__proto__:null,default:Ii},[$n]);/**
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
 */const Vr=ki||yh,wh=Ii||vh;class kh extends nu{constructor(e){super(),this.wasm=e,this.dataIdNextNumber=1,this.wasm.tfjs.initWithThreadsCount(Ah),this.wasm.tfjs.getThreadsCount(),this.dataIdMap=new ru(this,ds())}write(e,r,i){const s={id:this.dataIdNextNumber++};return this.move(s,e,r,i,1),s}numDataIds(){return this.dataIdMap.numDataIds()}async time(e){const r=Cr();return e(),{kernelMs:Cr()-r}}move(e,r,i,s,t){const u=this.dataIdNextNumber++;if(s==="string"){const n=r;this.dataIdMap.set(e,{id:u,stringBytes:n,shape:i,dtype:s,memoryOffset:null,refCount:t});return}const p=ee(i),h=p*Or(s),d=this.wasm._malloc(h);this.dataIdMap.set(e,{id:u,memoryOffset:d,shape:i,dtype:s,refCount:t}),this.wasm.tfjs.registerTensor(u,p,d),r!=null&&this.wasm.HEAPU8.set(new Uint8Array(r.buffer,r.byteOffset,h),d)}async read(e){return this.readSync(e)}readSync(e,r,i){const{memoryOffset:s,dtype:t,shape:u,stringBytes:p}=this.dataIdMap.get(e);if(t==="string")return(r==null||r===0)&&(i==null||i>=p.length)?p:p.slice(r,i);r=r||0,i=i||ee(u);const h=Or(t),d=this.wasm.HEAPU8.slice(s+r*h,s+i*h);return xh(d.buffer,t)}disposeData(e,r=!1){if(this.dataIdMap.has(e)){const i=this.dataIdMap.get(e);if(i.refCount--,!r&&i.refCount>0)return!1;this.wasm._free(i.memoryOffset),this.wasm.tfjs.disposeData(i.id),this.dataIdMap.delete(e)}return!0}refCount(e){return this.dataIdMap.has(e)?this.dataIdMap.get(e).refCount:0}incRef(e){const r=this.dataIdMap.get(e);r!=null&&r.refCount++}floatPrecision(){return 32}getMemoryOffset(e){return this.dataIdMap.get(e).memoryOffset}dispose(){this.wasm.tfjs.dispose(),"PThread"in this.wasm&&this.wasm.PThread.terminateAllThreads(),this.wasm=null}memory(){return{unreliable:!1}}makeOutput(e,r,i){let s;if(i==null)s=this.write(null,e,r);else{const t=this.dataIdNextNumber++;s={id:t},this.dataIdMap.set(s,{id:t,memoryOffset:i,shape:e,dtype:r,refCount:1});const u=ee(e);this.wasm.tfjs.registerTensor(t,u,i)}return{dataId:s,shape:e,dtype:r}}typedArrayFromHeap({shape:e,dtype:r,dataId:i}){const s=this.wasm.HEAPU8.buffer,{memoryOffset:t}=this.dataIdMap.get(i),u=ee(e);switch(r){case"float32":return new Float32Array(s,t,u);case"int32":return new Int32Array(s,t,u);case"bool":return new Uint8Array(s,t,u);default:throw new Error(`Unknown dtype ${r}`)}}}function Ih(a,e,r){let i="tfjs-backend-wasm.wasm";return a&&e?i="tfjs-backend-wasm-threaded-simd.wasm":a&&(i="tfjs-backend-wasm-simd.wasm"),ar!=null&&ar[i]!=null?ar[i]:r+i}async function Sh(){const[a,e]=await Promise.all([Hn().getAsync("WASM_HAS_SIMD_SUPPORT"),Hn().getAsync("WASM_HAS_MULTITHREAD_SUPPORT")]);return new Promise((r,i)=>{const s={};s.locateFile=(p,h)=>{if(p.endsWith(".worker.js")){const d=bh.replace(/\n/g,"\\n"),n=new Blob([d],{type:"application/javascript"});return URL.createObjectURL(n)}return p.endsWith(".wasm")?Ih(a,e,h):h+p};let t=!1;s.onAbort=()=>{if(t||sr)return;sr=!0,i({message:"Make sure the server can serve the `.wasm` file relative to the bundled js file. For more details see https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers"})};let u;e&&a&&Mh==null?(s.mainScriptUrlOrBlob=new Blob(["var WasmBackendModuleThreadedSimd = "+Vr.toString()],{type:"text/javascript"}),u=Vr(s)):u=wh(s),u.then(p=>{t=!0,sr=!1;const h=null;p.tfjs={init:p.cwrap("init",null,[]),initWithThreadsCount:p.cwrap("init_with_threads_count",null,["number"]),getThreadsCount:p.cwrap("get_threads_count","number",[]),registerTensor:p.cwrap("register_tensor",null,["number","number","number"]),disposeData:p.cwrap("dispose_data",h,["number"]),dispose:p.cwrap("dispose",h,[])},r({wasm:p})}).catch(i)})}function xh(a,e){switch(e){case"float32":return new Float32Array(a);case"int32":return new Int32Array(a);case"bool":return new Uint8Array(a);default:throw new Error(`Unknown dtype ${e}`)}}let Mh=null,ar={},sr=!1,Ah=-1;/**
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
 */const Th=2;au("wasm",async()=>{const{wasm:a}=await Sh();return new kh(a)},Th);var Rh={};(function(){var a;function e(o){var l=0;return function(){return l<o.length?{done:!1,value:o[l++]}:{done:!0}}}var r=typeof Object.defineProperties=="function"?Object.defineProperty:function(o,l,c){return o==Array.prototype||o==Object.prototype||(o[l]=c.value),o};function i(o){o=[typeof globalThis=="object"&&globalThis,o,typeof window=="object"&&window,typeof self=="object"&&self,typeof An=="object"&&An];for(var l=0;l<o.length;++l){var c=o[l];if(c&&c.Math==Math)return c}throw Error("Cannot find global object")}var s=i(this);function t(o,l){if(l)e:{var c=s;o=o.split(".");for(var v=0;v<o.length-1;v++){var M=o[v];if(!(M in c))break e;c=c[M]}o=o[o.length-1],v=c[o],l=l(v),l!=v&&l!=null&&r(c,o,{configurable:!0,writable:!0,value:l})}}t("Symbol",function(o){function l(C){if(this instanceof l)throw new TypeError("Symbol is not a constructor");return new c(v+(C||"")+"_"+M++,C)}function c(C,T){this.g=C,r(this,"description",{configurable:!0,writable:!0,value:T})}if(o)return o;c.prototype.toString=function(){return this.g};var v="jscomp_symbol_"+(1e9*Math.random()>>>0)+"_",M=0;return l}),t("Symbol.iterator",function(o){if(o)return o;o=Symbol("Symbol.iterator");for(var l="Array Int8Array Uint8Array Uint8ClampedArray Int16Array Uint16Array Int32Array Uint32Array Float32Array Float64Array".split(" "),c=0;c<l.length;c++){var v=s[l[c]];typeof v=="function"&&typeof v.prototype[o]!="function"&&r(v.prototype,o,{configurable:!0,writable:!0,value:function(){return u(e(this))}})}return o});function u(o){return o={next:o},o[Symbol.iterator]=function(){return this},o}function p(o){var l=typeof Symbol<"u"&&Symbol.iterator&&o[Symbol.iterator];return l?l.call(o):{next:e(o)}}function h(o){if(!(o instanceof Array)){o=p(o);for(var l,c=[];!(l=o.next()).done;)c.push(l.value);o=c}return o}var d=typeof Object.create=="function"?Object.create:function(o){function l(){}return l.prototype=o,new l},n;if(typeof Object.setPrototypeOf=="function")n=Object.setPrototypeOf;else{var f;e:{var _={a:!0},m={};try{m.__proto__=_,f=m.a;break e}catch{}f=!1}n=f?function(o,l){if(o.__proto__=l,o.__proto__!==l)throw new TypeError(o+" is not extensible");return o}:null}var y=n;function b(o,l){if(o.prototype=d(l.prototype),o.prototype.constructor=o,y)y(o,l);else for(var c in l)if(c!="prototype")if(Object.defineProperties){var v=Object.getOwnPropertyDescriptor(l,c);v&&Object.defineProperty(o,c,v)}else o[c]=l[c];o.ea=l.prototype}function k(){this.l=!1,this.i=null,this.h=void 0,this.g=1,this.s=this.m=0,this.j=null}function S(o){if(o.l)throw new TypeError("Generator is already running");o.l=!0}k.prototype.o=function(o){this.h=o};function I(o,l){o.j={U:l,V:!0},o.g=o.m||o.s}k.prototype.return=function(o){this.j={return:o},this.g=this.s};function w(o,l,c){return o.g=c,{value:l}}function A(o){this.g=new k,this.h=o}function O(o,l){S(o.g);var c=o.g.i;return c?F(o,"return"in c?c.return:function(v){return{value:v,done:!0}},l,o.g.return):(o.g.return(l),P(o))}function F(o,l,c,v){try{var M=l.call(o.g.i,c);if(!(M instanceof Object))throw new TypeError("Iterator result "+M+" is not an object");if(!M.done)return o.g.l=!1,M;var C=M.value}catch(T){return o.g.i=null,I(o.g,T),P(o)}return o.g.i=null,v.call(o.g,C),P(o)}function P(o){for(;o.g.g;)try{var l=o.h(o.g);if(l)return o.g.l=!1,{value:l.value,done:!1}}catch(c){o.g.h=void 0,I(o.g,c)}if(o.g.l=!1,o.g.j){if(l=o.g.j,o.g.j=null,l.V)throw l.U;return{value:l.return,done:!0}}return{value:void 0,done:!0}}function V(o){this.next=function(l){return S(o.g),o.g.i?l=F(o,o.g.i.next,l,o.g.o):(o.g.o(l),l=P(o)),l},this.throw=function(l){return S(o.g),o.g.i?l=F(o,o.g.i.throw,l,o.g.o):(I(o.g,l),l=P(o)),l},this.return=function(l){return O(o,l)},this[Symbol.iterator]=function(){return this}}function D(o,l){return l=new V(new A(l)),y&&o.prototype&&y(l,o.prototype),l}function j(o,l){o instanceof String&&(o+="");var c=0,v=!1,M={next:function(){if(!v&&c<o.length){var C=c++;return{value:l(C,o[C]),done:!1}}return v=!0,{done:!0,value:void 0}}};return M[Symbol.iterator]=function(){return M},M}var G=typeof Object.assign=="function"?Object.assign:function(o,l){for(var c=1;c<arguments.length;c++){var v=arguments[c];if(v)for(var M in v)Object.prototype.hasOwnProperty.call(v,M)&&(o[M]=v[M])}return o};t("Object.assign",function(o){return o||G}),t("Promise",function(o){function l(T){this.h=0,this.i=void 0,this.g=[],this.o=!1;var E=this.j();try{T(E.resolve,E.reject)}catch(B){E.reject(B)}}function c(){this.g=null}function v(T){return T instanceof l?T:new l(function(E){E(T)})}if(o)return o;c.prototype.h=function(T){if(this.g==null){this.g=[];var E=this;this.i(function(){E.l()})}this.g.push(T)};var M=s.setTimeout;c.prototype.i=function(T){M(T,0)},c.prototype.l=function(){for(;this.g&&this.g.length;){var T=this.g;this.g=[];for(var E=0;E<T.length;++E){var B=T[E];T[E]=null;try{B()}catch(z){this.j(z)}}}this.g=null},c.prototype.j=function(T){this.i(function(){throw T})},l.prototype.j=function(){function T(z){return function(Z){B||(B=!0,z.call(E,Z))}}var E=this,B=!1;return{resolve:T(this.C),reject:T(this.l)}},l.prototype.C=function(T){if(T===this)this.l(new TypeError("A Promise cannot resolve to itself"));else if(T instanceof l)this.F(T);else{e:switch(typeof T){case"object":var E=T!=null;break e;case"function":E=!0;break e;default:E=!1}E?this.u(T):this.m(T)}},l.prototype.u=function(T){var E=void 0;try{E=T.then}catch(B){this.l(B);return}typeof E=="function"?this.G(E,T):this.m(T)},l.prototype.l=function(T){this.s(2,T)},l.prototype.m=function(T){this.s(1,T)},l.prototype.s=function(T,E){if(this.h!=0)throw Error("Cannot settle("+T+", "+E+"): Promise already settled in state"+this.h);this.h=T,this.i=E,this.h===2&&this.D(),this.A()},l.prototype.D=function(){var T=this;M(function(){if(T.B()){var E=s.console;typeof E<"u"&&E.error(T.i)}},1)},l.prototype.B=function(){if(this.o)return!1;var T=s.CustomEvent,E=s.Event,B=s.dispatchEvent;return typeof B>"u"?!0:(typeof T=="function"?T=new T("unhandledrejection",{cancelable:!0}):typeof E=="function"?T=new E("unhandledrejection",{cancelable:!0}):(T=s.document.createEvent("CustomEvent"),T.initCustomEvent("unhandledrejection",!1,!0,T)),T.promise=this,T.reason=this.i,B(T))},l.prototype.A=function(){if(this.g!=null){for(var T=0;T<this.g.length;++T)C.h(this.g[T]);this.g=null}};var C=new c;return l.prototype.F=function(T){var E=this.j();T.J(E.resolve,E.reject)},l.prototype.G=function(T,E){var B=this.j();try{T.call(E,B.resolve,B.reject)}catch(z){B.reject(z)}},l.prototype.then=function(T,E){function B(se,te){return typeof se=="function"?function(K){try{z(se(K))}catch(Y){Z(Y)}}:te}var z,Z,ve=new l(function(se,te){z=se,Z=te});return this.J(B(T,z),B(E,Z)),ve},l.prototype.catch=function(T){return this.then(void 0,T)},l.prototype.J=function(T,E){function B(){switch(z.h){case 1:T(z.i);break;case 2:E(z.i);break;default:throw Error("Unexpected state: "+z.h)}}var z=this;this.g==null?C.h(B):this.g.push(B),this.o=!0},l.resolve=v,l.reject=function(T){return new l(function(E,B){B(T)})},l.race=function(T){return new l(function(E,B){for(var z=p(T),Z=z.next();!Z.done;Z=z.next())v(Z.value).J(E,B)})},l.all=function(T){var E=p(T),B=E.next();return B.done?v([]):new l(function(z,Z){function ve(K){return function(Y){se[K]=Y,te--,te==0&&z(se)}}var se=[],te=0;do se.push(void 0),te++,v(B.value).J(ve(se.length-1),Z),B=E.next();while(!B.done)})},l}),t("Object.is",function(o){return o||function(l,c){return l===c?l!==0||1/l===1/c:l!==l&&c!==c}}),t("Array.prototype.includes",function(o){return o||function(l,c){var v=this;v instanceof String&&(v=String(v));var M=v.length;for(c=c||0,0>c&&(c=Math.max(c+M,0));c<M;c++){var C=v[c];if(C===l||Object.is(C,l))return!0}return!1}}),t("String.prototype.includes",function(o){return o||function(l,c){if(this==null)throw new TypeError("The 'this' value for String.prototype.includes must not be null or undefined");if(l instanceof RegExp)throw new TypeError("First argument to String.prototype.includes must not be a regular expression");return this.indexOf(l,c||0)!==-1}}),t("Array.prototype.keys",function(o){return o||function(){return j(this,function(l){return l})}});var X=this||self;function $(o,l){o=o.split(".");var c=X;o[0]in c||typeof c.execScript>"u"||c.execScript("var "+o[0]);for(var v;o.length&&(v=o.shift());)o.length||l===void 0?c[v]&&c[v]!==Object.prototype[v]?c=c[v]:c=c[v]={}:c[v]=l}function le(o,l){return l=String.fromCharCode.apply(null,l),o==null?l:o+l}var Q,ce=typeof TextDecoder<"u",ke,me=typeof TextEncoder<"u";function We(o){if(me)o=(ke||(ke=new TextEncoder)).encode(o);else{var l=void 0;l=l===void 0?!1:l;for(var c=0,v=new Uint8Array(3*o.length),M=0;M<o.length;M++){var C=o.charCodeAt(M);if(128>C)v[c++]=C;else{if(2048>C)v[c++]=C>>6|192;else{if(55296<=C&&57343>=C){if(56319>=C&&M<o.length){var T=o.charCodeAt(++M);if(56320<=T&&57343>=T){C=1024*(C-55296)+T-56320+65536,v[c++]=C>>18|240,v[c++]=C>>12&63|128,v[c++]=C>>6&63|128,v[c++]=C&63|128;continue}else M--}if(l)throw Error("Found an unpaired surrogate");C=65533}v[c++]=C>>12|224,v[c++]=C>>6&63|128}v[c++]=C&63|128}}o=v.subarray(0,c)}return o}var qe={},Le=null;function ft(o,l){l===void 0&&(l=0),xt(),l=qe[l];for(var c=Array(Math.floor(o.length/3)),v=l[64]||"",M=0,C=0;M<o.length-2;M+=3){var T=o[M],E=o[M+1],B=o[M+2],z=l[T>>2];T=l[(T&3)<<4|E>>4],E=l[(E&15)<<2|B>>6],B=l[B&63],c[C++]=z+T+E+B}switch(z=0,B=v,o.length-M){case 2:z=o[M+1],B=l[(z&15)<<2]||v;case 1:o=o[M],c[C]=l[o>>2]+l[(o&3)<<4|z>>4]+B+v}return c.join("")}function Be(o){var l=o.length,c=3*l/4;c%3?c=Math.floor(c):"=.".indexOf(o[l-1])!=-1&&(c="=.".indexOf(o[l-2])!=-1?c-2:c-1);var v=new Uint8Array(c),M=0;return Ct(o,function(C){v[M++]=C}),v.subarray(0,M)}function Ct(o,l){function c(B){for(;v<o.length;){var z=o.charAt(v++),Z=Le[z];if(Z!=null)return Z;if(!/^[\s\xa0]*$/.test(z))throw Error("Unknown base64 encoding at char: "+z)}return B}xt();for(var v=0;;){var M=c(-1),C=c(0),T=c(64),E=c(64);if(E===64&&M===-1)break;l(M<<2|C>>4),T!=64&&(l(C<<4&240|T>>2),E!=64&&l(T<<6&192|E))}}function xt(){if(!Le){Le={};for(var o="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split(""),l=["+/=","+/","-_=","-_.","-_"],c=0;5>c;c++){var v=o.concat(l[c].split(""));qe[c]=v;for(var M=0;M<v.length;M++){var C=v[M];Le[C]===void 0&&(Le[C]=M)}}}}var zt=typeof Uint8Array.prototype.slice=="function",Ot;function Fe(o,l,c){return l===c?Ot||(Ot=new Uint8Array(0)):zt?o.slice(l,c):new Uint8Array(o.subarray(l,c))}var He=0,ut=0;function $t(o,l){l=l===void 0?{}:l,l=l.v===void 0?!1:l.v,this.h=null,this.g=this.i=this.j=0,this.l=!1,this.v=l,o&&bt(this,o)}function bt(o,l){l=l.constructor===Uint8Array?l:l.constructor===ArrayBuffer?new Uint8Array(l):l.constructor===Array?new Uint8Array(l):l.constructor===String?Be(l):l instanceof Uint8Array?new Uint8Array(l.buffer,l.byteOffset,l.byteLength):new Uint8Array(0),o.h=l,o.j=0,o.i=o.h.length,o.g=o.j}$t.prototype.reset=function(){this.g=this.j};function _t(o){var l=o.h,c=l[o.g],v=c&127;return 128>c?(o.g+=1,v):(c=l[o.g+1],v|=(c&127)<<7,128>c?(o.g+=2,v):(c=l[o.g+2],v|=(c&127)<<14,128>c?(o.g+=3,v):(c=l[o.g+3],v|=(c&127)<<21,128>c?(o.g+=4,v):(c=l[o.g+4],v|=(c&15)<<28,128>c?(o.g+=5,v>>>0):(o.g+=5,128<=l[o.g++]&&128<=l[o.g++]&&128<=l[o.g++]&&128<=l[o.g++]&&o.g++,v)))))}function Ne(o){var l=o.h[o.g],c=o.h[o.g+1],v=o.h[o.g+2],M=o.h[o.g+3];return o.g+=4,c=(l<<0|c<<8|v<<16|M<<24)>>>0,o=2*(c>>31)+1,l=c>>>23&255,c&=8388607,l==255?c?NaN:1/0*o:l==0?o*Math.pow(2,-149)*c:o*Math.pow(2,l-150)*(c+Math.pow(2,23))}var Pt=[];function Ce(){this.g=new Uint8Array(64),this.h=0}Ce.prototype.push=function(o){if(!(this.h+1<this.g.length)){var l=this.g;this.g=new Uint8Array(Math.ceil(1+2*this.g.length)),this.g.set(l)}this.g[this.h++]=o},Ce.prototype.length=function(){return this.h},Ce.prototype.end=function(){var o=this.g,l=this.h;return this.h=0,Fe(o,0,l)};function nt(o,l){for(;127<l;)o.push(l&127|128),l>>>=7;o.push(l)}function Mt(o){var l={},c=l.N===void 0?!1:l.N;this.o={v:l.v===void 0?!1:l.v},this.N=c,l=this.o,Pt.length?(c=Pt.pop(),l&&(c.v=l.v),o&&bt(c,o),o=c):o=new $t(o,l),this.g=o,this.m=this.g.g,this.h=this.i=this.l=-1,this.j=!1}Mt.prototype.reset=function(){this.g.reset(),this.h=this.l=-1};function mt(o){var l=o.g;if((l=l.g==l.i)||(l=o.j)||(l=o.g,l=l.l||0>l.g||l.g>l.i),l)return!1;o.m=o.g.g,l=_t(o.g);var c=l&7;return c!=0&&c!=5&&c!=1&&c!=2&&c!=3&&c!=4?(o.j=!0,!1):(o.i=l,o.l=l>>>3,o.h=c,!0)}function lt(o){switch(o.h){case 0:if(o.h!=0)lt(o);else{for(o=o.g;o.h[o.g]&128;)o.g++;o.g++}break;case 1:o.h!=1?lt(o):(o=o.g,o.g+=8);break;case 2:if(o.h!=2)lt(o);else{var l=_t(o.g);o=o.g,o.g+=l}break;case 5:o.h!=5?lt(o):(o=o.g,o.g+=4);break;case 3:l=o.l;do{if(!mt(o)){o.j=!0;break}if(o.h==4){o.l!=l&&(o.j=!0);break}lt(o)}while(1);break;default:o.j=!0}}function Dt(o,l,c){var v=o.g.i,M=_t(o.g),C=o.g.g+M;if(o.g.i=C,c(l,o),c=C-o.g.g,c!==0)throw Error("Message parsing ended unexpectedly. Expected to read "+M+" bytes, instead read "+(M-c)+" bytes, either the data ended unexpectedly or the message misreported its own length");return o.g.g=C,o.g.i=v,l}function Gt(o){var l=_t(o.g);o=o.g;var c=o.g;o.g+=l,o=o.h;var v;if(ce)(v=Q)||(v=Q=new TextDecoder("utf-8",{fatal:!1})),v=v.decode(o.subarray(c,c+l));else{l=c+l;for(var M=[],C=null,T,E,B;c<l;)T=o[c++],128>T?M.push(T):224>T?c>=l?M.push(65533):(E=o[c++],194>T||(E&192)!==128?(c--,M.push(65533)):M.push((T&31)<<6|E&63)):240>T?c>=l-1?M.push(65533):(E=o[c++],(E&192)!==128||T===224&&160>E||T===237&&160<=E||((v=o[c++])&192)!==128?(c--,M.push(65533)):M.push((T&15)<<12|(E&63)<<6|v&63)):244>=T?c>=l-2?M.push(65533):(E=o[c++],(E&192)!==128||(T<<28)+(E-144)>>30||((v=o[c++])&192)!==128||((B=o[c++])&192)!==128?(c--,M.push(65533)):(T=(T&7)<<18|(E&63)<<12|(v&63)<<6|B&63,T-=65536,M.push((T>>10&1023)+55296,(T&1023)+56320))):M.push(65533),8192<=M.length&&(C=le(C,M),M.length=0);v=le(C,M)}return v}function qt(){this.h=[],this.i=0,this.g=new Ce}function gt(o,l){l.length!==0&&(o.h.push(l),o.i+=l.length)}function Kt(o){var l=o.i+o.g.length();if(l===0)return new Uint8Array(0);l=new Uint8Array(l);for(var c=o.h,v=c.length,M=0,C=0;C<v;C++){var T=c[C];T.length!==0&&(l.set(T,M),M+=T.length)}return c=o.g,v=c.h,v!==0&&(l.set(c.g.subarray(0,v),M),c.h=0),o.h=[l],l}function Qe(o,l,c){if(c!=null){nt(o.g,8*l+5),o=o.g;var v=c;v=(c=0>v?1:0)?-v:v,v===0?0<1/v?He=ut=0:(ut=0,He=2147483648):isNaN(v)?(ut=0,He=2147483647):34028234663852886e22<v?(ut=0,He=(c<<31|2139095040)>>>0):11754943508222875e-54>v?(v=Math.round(v/Math.pow(2,-149)),ut=0,He=(c<<31|v)>>>0):(l=Math.floor(Math.log(v)/Math.LN2),v*=Math.pow(2,-l),v=Math.round(8388608*v),16777216<=v&&++l,ut=0,He=(c<<31|l+127<<23|v&8388607)>>>0),c=He,o.push(c>>>0&255),o.push(c>>>8&255),o.push(c>>>16&255),o.push(c>>>24&255)}}var dt=typeof Uint8Array=="function";function vt(o,l,c){if(o!=null)return typeof o=="object"?dt&&o instanceof Uint8Array?c(o):cn(o,l,c):l(o)}function cn(o,l,c){if(Array.isArray(o)){for(var v=Array(o.length),M=0;M<o.length;M++)v[M]=vt(o[M],l,c);return Array.isArray(o)&&o.W&&Tt(v),v}v={};for(M in o)v[M]=vt(o[M],l,c);return v}function vn(o){return typeof o=="number"?isFinite(o)?o:String(o):o}var At={W:{value:!0,configurable:!0}};function Tt(o){return Array.isArray(o)&&!Object.isFrozen(o)&&Object.defineProperties(o,At),o}var Xt;function Ue(o,l,c){var v=Xt;Xt=null,o||(o=v),v=this.constructor.ca,o||(o=v?[v]:[]),this.j=v?0:-1,this.i=null,this.g=o;e:{if(v=this.g.length,o=v-1,v&&(v=this.g[o],!(v===null||typeof v!="object"||Array.isArray(v)||dt&&v instanceof Uint8Array))){this.l=o-this.j,this.h=v;break e}l!==void 0&&-1<l?(this.l=Math.max(l,o+1-this.j),this.h=null):this.l=Number.MAX_VALUE}if(c)for(l=0;l<c.length;l++)o=c[l],o<this.l?(o+=this.j,(v=this.g[o])?Tt(v):this.g[o]=Oe):(Yt(this),(v=this.h[o])?Tt(v):this.h[o]=Oe)}var Oe=Object.freeze(Tt([]));function Yt(o){var l=o.l+o.j;o.g[l]||(o.h=o.g[l]={})}function Me(o,l,c){return l===-1?null:c!==void 0&&c||l>=o.l?o.h?o.h[l]:void 0:o.g[l+o.j]}function Qt(o){var l=l===void 0?!1:l,c=Me(o,1,l);return c==null&&(c=Oe),c===Oe&&(c=Tt([]),pe(o,1,c,l)),c}function Je(o,l,c){return o=Me(o,l),o=o==null?o:+o,o??(c===void 0?0:c)}function pe(o,l,c,v){v!==void 0&&v||l>=o.l?(Yt(o),o.h[l]=c):o.g[l+o.j]=c}function wt(o,l){o.i||(o.i={});var c=o.i[1];if(!c){var v=Qt(o);c=[];for(var M=0;M<v.length;M++)c[M]=new l(v[M]);o.i[1]=c}return c}function Jt(o,l,c,v){var M=wt(o,c);l=l||new c,o=Qt(o),v!=null?(M.splice(v,0,l),o.splice(v,0,rt(l))):(M.push(l),o.push(rt(l)))}Ue.prototype.toJSON=function(){var o=rt(this);return cn(o,vn,ft)};function rt(o,l){if(o.i)for(var c in o.i){var v=o.i[c];if(Array.isArray(v))for(var M=0;M<v.length;M++)v[M]&&rt(v[M]);else v&&rt(v)}return o.g}Ue.prototype.toString=function(){return rt(this).toString()};function kt(o,l){return o=Me(o,l),o??0}function Zt(o,l){return o=Me(o,l),o??""}function R(o,l){if(o=o.m){gt(l,l.g.end());for(var c=0;c<o.length;c++)gt(l,o[c])}}function L(o,l){if(l.h==4)return!1;var c=l.m;return lt(l),l.N||(l=Fe(l.g.h,c,l.g.g),(c=o.m)?c.push(l):o.m=[l]),!0}function U(o,l){var c=void 0;return new(c||(c=Promise))(function(v,M){function C(B){try{E(l.next(B))}catch(z){M(z)}}function T(B){try{E(l.throw(B))}catch(z){M(z)}}function E(B){B.done?v(B.value):new c(function(z){z(B.value)}).then(C,T)}E((l=l.apply(o,void 0)).next())})}function W(o){Ue.call(this,o)}b(W,Ue);function Ie(o,l){for(;mt(l);)switch(l.i){case 8:var c=_t(l.g);pe(o,1,c);break;case 21:c=Ne(l.g),pe(o,2,c);break;case 26:c=Gt(l),pe(o,3,c);break;case 34:c=Gt(l),pe(o,4,c);break;default:if(!L(o,l))return o}return o}function Se(o){Ue.call(this,o,-1,ue)}b(Se,Ue),Se.prototype.addClassification=function(o,l){return Jt(this,o,W,l),this};var ue=[1];function J(o){Ue.call(this,o)}b(J,Ue);function Ae(o,l){for(;mt(l);)switch(l.i){case 13:var c=Ne(l.g);pe(o,1,c);break;case 21:c=Ne(l.g),pe(o,2,c);break;case 29:c=Ne(l.g),pe(o,3,c);break;case 37:c=Ne(l.g),pe(o,4,c);break;case 45:c=Ne(l.g),pe(o,5,c);break;default:if(!L(o,l))return o}return o}function Ke(o){Ue.call(this,o,-1,yt)}b(Ke,Ue);var yt=[1];function Lt(o){Ue.call(this,o)}b(Lt,Ue);function Bt(o,l,c){if(c=o.createShader(c===0?o.VERTEX_SHADER:o.FRAGMENT_SHADER),o.shaderSource(c,l),o.compileShader(c),!o.getShaderParameter(c,o.COMPILE_STATUS))throw Error(`Could not compile WebGL shader.

`+o.getShaderInfoLog(c));return c}function wn(o){return wt(o,W).map(function(l){return{index:kt(l,1),Y:Je(l,2),label:Me(l,3)!=null?Zt(l,3):void 0,displayName:Me(l,4)!=null?Zt(l,4):void 0}})}function ze(o){return{x:Je(o,1),y:Je(o,2),z:Je(o,3),visibility:Me(o,4)!=null?Je(o,4):void 0}}function ct(o){e:{var l=new Ke;for(o=new Mt(o);mt(o);)switch(o.i){case 10:var c=Dt(o,new J,Ae);Jt(l,c,J,void 0);break;default:if(!L(l,o))break e}}return wt(l,J).map(ze)}function Rt(o,l){this.h=o,this.g=l,this.l=0}function Pn(o,l,c){return Kn(o,l),typeof o.g.canvas.transferToImageBitmap=="function"?Promise.resolve(o.g.canvas.transferToImageBitmap()):c?Promise.resolve(o.g.canvas):typeof createImageBitmap=="function"?createImageBitmap(o.g.canvas):(o.i===void 0&&(o.i=document.createElement("canvas")),new Promise(function(v){o.i.height=o.g.canvas.height,o.i.width=o.g.canvas.width,o.i.getContext("2d",{}).drawImage(o.g.canvas,0,0,o.g.canvas.width,o.g.canvas.height),v(o.i)}))}function Kn(o,l){var c=o.g;if(o.m===void 0){var v=Bt(c,`
  attribute vec2 aVertex;
  attribute vec2 aTex;
  varying vec2 vTex;
  void main(void) {
    gl_Position = vec4(aVertex, 0.0, 1.0);
    vTex = aTex;
  }`,0),M=Bt(c,`
  precision mediump float;
  varying vec2 vTex;
  uniform sampler2D sampler0;
  void main(){
    gl_FragColor = texture2D(sampler0, vTex);
  }`,1),C=c.createProgram();if(c.attachShader(C,v),c.attachShader(C,M),c.linkProgram(C),!c.getProgramParameter(C,c.LINK_STATUS))throw Error(`Could not compile WebGL program.

`+c.getProgramInfoLog(C));v=o.m=C,c.useProgram(v),M=c.getUniformLocation(v,"sampler0"),o.j={I:c.getAttribLocation(v,"aVertex"),H:c.getAttribLocation(v,"aTex"),da:M},o.s=c.createBuffer(),c.bindBuffer(c.ARRAY_BUFFER,o.s),c.enableVertexAttribArray(o.j.I),c.vertexAttribPointer(o.j.I,2,c.FLOAT,!1,0,0),c.bufferData(c.ARRAY_BUFFER,new Float32Array([-1,-1,-1,1,1,1,1,-1]),c.STATIC_DRAW),c.bindBuffer(c.ARRAY_BUFFER,null),o.o=c.createBuffer(),c.bindBuffer(c.ARRAY_BUFFER,o.o),c.enableVertexAttribArray(o.j.H),c.vertexAttribPointer(o.j.H,2,c.FLOAT,!1,0,0),c.bufferData(c.ARRAY_BUFFER,new Float32Array([0,1,0,0,1,0,1,1]),c.STATIC_DRAW),c.bindBuffer(c.ARRAY_BUFFER,null),c.uniform1i(M,0)}v=o.j,c.useProgram(o.m),c.canvas.width=l.width,c.canvas.height=l.height,c.viewport(0,0,l.width,l.height),c.activeTexture(c.TEXTURE0),o.h.bindTexture2d(l.glName),c.enableVertexAttribArray(v.I),c.bindBuffer(c.ARRAY_BUFFER,o.s),c.vertexAttribPointer(v.I,2,c.FLOAT,!1,0,0),c.enableVertexAttribArray(v.H),c.bindBuffer(c.ARRAY_BUFFER,o.o),c.vertexAttribPointer(v.H,2,c.FLOAT,!1,0,0),c.bindFramebuffer(c.DRAW_FRAMEBUFFER?c.DRAW_FRAMEBUFFER:c.FRAMEBUFFER,null),c.clearColor(0,0,0,0),c.clear(c.COLOR_BUFFER_BIT),c.colorMask(!0,!0,!0,!0),c.drawArrays(c.TRIANGLE_FAN,0,4),c.disableVertexAttribArray(v.I),c.disableVertexAttribArray(v.H),c.bindBuffer(c.ARRAY_BUFFER,null),o.h.bindTexture2d(0)}function Xn(o){this.g=o}var kn=new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,10,9,1,7,0,65,0,253,15,26,11]);function Yn(o,l){return l+o}function Dn(o,l){window[o]=l}function Qn(o){var l=document.createElement("script");return l.setAttribute("src",o),l.setAttribute("crossorigin","anonymous"),new Promise(function(c){l.addEventListener("load",function(){c()},!1),l.addEventListener("error",function(){c()},!1),document.body.appendChild(l)})}function en(){return U(this,function o(){return D(o,function(l){switch(l.g){case 1:return l.m=2,w(l,WebAssembly.instantiate(kn),4);case 4:l.g=3,l.m=0;break;case 2:return l.m=0,l.j=null,l.return(!1);case 3:return l.return(!0)}})})}function In(o){if(this.g=o,this.listeners={},this.j={},this.F={},this.m={},this.s={},this.G=this.o=this.R=!0,this.C=Promise.resolve(),this.P="",this.B={},this.locateFile=o&&o.locateFile||Yn,typeof window=="object")var l=window.location.pathname.toString().substring(0,window.location.pathname.toString().lastIndexOf("/"))+"/";else if(typeof location<"u")l=location.pathname.toString().substring(0,location.pathname.toString().lastIndexOf("/"))+"/";else throw Error("solutions can only be loaded on a web page or in a web worker");if(this.S=l,o.options){l=p(Object.keys(o.options));for(var c=l.next();!c.done;c=l.next()){c=c.value;var v=o.options[c].default;v!==void 0&&(this.j[c]=typeof v=="function"?v():v)}}}a=In.prototype,a.close=function(){return this.i&&this.i.delete(),Promise.resolve()};function Jn(o,l){return o.g.files===void 0?[]:typeof o.g.files=="function"?o.g.files(l):o.g.files}function Ln(o){return U(o,function l(){var c=this,v,M,C,T,E,B,z,Z,ve,se,te;return D(l,function(K){switch(K.g){case 1:return v=c,c.R?(M=Jn(c,c.j),w(K,en(),2)):K.return();case 2:if(C=K.h,typeof window=="object")return Dn("createMediapipeSolutionsWasm",{locateFile:c.locateFile}),Dn("createMediapipeSolutionsPackedAssets",{locateFile:c.locateFile}),B=M.filter(function(Y){return Y.data!==void 0}),z=M.filter(function(Y){return Y.data===void 0}),Z=Promise.all(B.map(function(Y){var ie=dn(v,Y.url);if(Y.path!==void 0){var de=Y.path;ie=ie.then(function(Pe){return v.overrideFile(de,Pe),Promise.resolve(Pe)})}return ie})),ve=Promise.all(z.map(function(Y){return Y.simd===void 0||Y.simd&&C||!Y.simd&&!C?Qn(v.locateFile(Y.url,v.S)):Promise.resolve()})).then(function(){return U(v,function Y(){var ie,de,Pe=this;return D(Y,function(we){if(we.g==1)return ie=window.createMediapipeSolutionsWasm,de=window.createMediapipeSolutionsPackedAssets,w(we,ie(de),2);Pe.h=we.h,we.g=0})})}),se=function(){return U(v,function Y(){var ie=this;return D(Y,function(de){return ie.g.graph&&ie.g.graph.url?de=w(de,dn(ie,ie.g.graph.url),0):(de.g=0,de=void 0),de})})}(),w(K,Promise.all([ve,Z,se]),7);if(typeof importScripts!="function")throw Error("solutions can only be loaded on a web page or in a web worker");return T=M.filter(function(Y){return Y.simd===void 0||Y.simd&&C||!Y.simd&&!C}).map(function(Y){return v.locateFile(Y.url,v.S)}),importScripts.apply(null,h(T)),w(K,createMediapipeSolutionsWasm(Module),6);case 6:c.h=K.h,c.l=new OffscreenCanvas(1,1),c.h.canvas=c.l,E=c.h.GL.createContext(c.l,{antialias:!1,alpha:!1,ba:typeof WebGL2RenderingContext<"u"?2:1}),c.h.GL.makeContextCurrent(E),K.g=4;break;case 7:if(c.l=document.createElement("canvas"),te=c.l.getContext("webgl2",{}),!te&&(te=c.l.getContext("webgl",{}),!te))return alert("Failed to create WebGL canvas context when passing video frame."),K.return();c.D=te,c.h.canvas=c.l,c.h.createContext(c.l,!0,!0,{});case 4:c.i=new c.h.SolutionWasm,c.R=!1,K.g=0}})})}function Zn(o){return U(o,function l(){var c=this,v,M,C,T,E,B,z,Z;return D(l,function(ve){if(ve.g==1){if(c.g.graph&&c.g.graph.url&&c.P===c.g.graph.url)return ve.return();if(c.o=!0,!c.g.graph||!c.g.graph.url){ve.g=2;return}return c.P=c.g.graph.url,w(ve,dn(c,c.g.graph.url),3)}for(ve.g!=2&&(v=ve.h,c.i.loadGraph(v)),M=p(Object.keys(c.B)),C=M.next();!C.done;C=M.next())T=C.value,c.i.overrideFile(T,c.B[T]);if(c.B={},c.g.listeners)for(E=p(c.g.listeners),B=E.next();!B.done;B=E.next())z=B.value,Sn(c,z);Z=c.j,c.j={},c.setOptions(Z),ve.g=0})})}a.reset=function(){return U(this,function o(){var l=this;return D(o,function(c){l.i&&(l.i.reset(),l.m={},l.s={}),c.g=0})})},a.setOptions=function(o,l){var c=this;if(l=l||this.g.options){for(var v=[],M=[],C={},T=p(Object.keys(o)),E=T.next();!E.done;C={K:C.K,L:C.L},E=T.next()){var B=E.value;B in this.j&&this.j[B]===o[B]||(this.j[B]=o[B],E=l[B],E!==void 0&&(E.onChange&&(C.K=E.onChange,C.L=o[B],v.push(function(z){return function(){return U(c,function Z(){var ve,se=this;return D(Z,function(te){if(te.g==1)return w(te,z.K(z.L),2);ve=te.h,ve===!0&&(se.o=!0),te.g=0})})}}(C))),E.graphOptionXref&&(B={valueNumber:E.type===1?o[B]:0,valueBoolean:E.type===0?o[B]:!1,valueString:E.type===2?o[B]:""},E=Object.assign(Object.assign(Object.assign({},{calculatorName:"",calculatorIndex:0}),E.graphOptionXref),B),M.push(E))))}(v.length!==0||M.length!==0)&&(this.o=!0,this.A=(this.A===void 0?[]:this.A).concat(M),this.u=(this.u===void 0?[]:this.u).concat(v))}};function pn(o){return U(o,function l(){var c=this,v,M,C,T,E,B,z;return D(l,function(Z){switch(Z.g){case 1:if(!c.o)return Z.return();if(!c.u){Z.g=2;break}v=p(c.u),M=v.next();case 3:if(M.done){Z.g=5;break}return C=M.value,w(Z,C(),4);case 4:M=v.next(),Z.g=3;break;case 5:c.u=void 0;case 2:if(c.A){for(T=new c.h.GraphOptionChangeRequestList,E=p(c.A),B=E.next();!B.done;B=E.next())z=B.value,T.push_back(z);c.i.changeOptions(T),T.delete(),c.A=void 0}c.o=!1,Z.g=0}})})}a.initialize=function(){return U(this,function o(){var l=this;return D(o,function(c){return c.g==1?w(c,Ln(l),2):c.g!=3?w(c,Zn(l),3):w(c,pn(l),0)})})};function dn(o,l){return U(o,function c(){var v=this,M,C;return D(c,function(T){return l in v.F?T.return(v.F[l]):(M=v.locateFile(l,""),C=fetch(M).then(function(E){return E.arrayBuffer()}),v.F[l]=C,T.return(C))})})}a.overrideFile=function(o,l){this.i?this.i.overrideFile(o,l):this.B[o]=l},a.clearOverriddenFiles=function(){this.B={},this.i&&this.i.clearOverriddenFiles()},a.send=function(o,l){return U(this,function c(){var v=this,M,C,T,E,B,z,Z,ve,se;return D(c,function(te){switch(te.g){case 1:return v.g.inputs?(M=1e3*(l??performance.now()),w(te,v.C,2)):te.return();case 2:return w(te,v.initialize(),3);case 3:for(C=new v.h.PacketDataList,T=p(Object.keys(o)),E=T.next();!E.done;E=T.next())if(B=E.value,z=v.g.inputs[B]){e:{var K=v,Y=o[B];switch(z.type){case"video":var ie=K.m[z.stream];if(ie||(ie=new Rt(K.h,K.D),K.m[z.stream]=ie),K=ie,K.l===0&&(K.l=K.h.createTexture()),typeof HTMLVideoElement<"u"&&Y instanceof HTMLVideoElement){var de=Y.videoWidth;ie=Y.videoHeight}else typeof HTMLImageElement<"u"&&Y instanceof HTMLImageElement?(de=Y.naturalWidth,ie=Y.naturalHeight):(de=Y.width,ie=Y.height);ie={glName:K.l,width:de,height:ie},de=K.g,de.canvas.width=ie.width,de.canvas.height=ie.height,de.activeTexture(de.TEXTURE0),K.h.bindTexture2d(K.l),de.texImage2D(de.TEXTURE_2D,0,de.RGBA,de.RGBA,de.UNSIGNED_BYTE,Y),K.h.bindTexture2d(0),K=ie;break e;case"detections":for(ie=K.m[z.stream],ie||(ie=new Xn(K.h),K.m[z.stream]=ie),K=ie,K.data||(K.data=new K.g.DetectionListData),K.data.reset(Y.length),ie=0;ie<Y.length;++ie){de=Y[ie];var Pe=K.data,we=Pe.setBoundingBox,je=ie,Te=de.T,ne=new Lt;pe(ne,1,Te.Z),pe(ne,2,Te.$),pe(ne,3,Te.height),pe(ne,4,Te.width),pe(ne,5,Te.rotation),pe(ne,6,Te.X);var fe=Te=new qt;Qe(fe,1,Me(ne,1)),Qe(fe,2,Me(ne,2)),Qe(fe,3,Me(ne,3)),Qe(fe,4,Me(ne,4)),Qe(fe,5,Me(ne,5));var he=Me(ne,6);if(he!=null&&he!=null){nt(fe.g,48);var oe=fe.g,ge=he;he=0>ge,ge=Math.abs(ge);var g=ge>>>0;for(ge=Math.floor((ge-g)/4294967296),ge>>>=0,he&&(ge=~ge>>>0,g=(~g>>>0)+1,4294967295<g&&(g=0,ge++,4294967295<ge&&(ge=0))),He=g,ut=ge,he=He,g=ut;0<g||127<he;)oe.push(he&127|128),he=(he>>>7|g<<25)>>>0,g>>>=7;oe.push(he)}if(R(ne,fe),Te=Kt(Te),we.call(Pe,je,Te),de.O)for(Pe=0;Pe<de.O.length;++Pe)ne=de.O[Pe],fe=!!ne.visibility,we=K.data,je=we.addNormalizedLandmark,Te=ie,ne=Object.assign(Object.assign({},ne),{visibility:fe?ne.visibility:0}),fe=new J,pe(fe,1,ne.x),pe(fe,2,ne.y),pe(fe,3,ne.z),ne.visibility&&pe(fe,4,ne.visibility),oe=ne=new qt,Qe(oe,1,Me(fe,1)),Qe(oe,2,Me(fe,2)),Qe(oe,3,Me(fe,3)),Qe(oe,4,Me(fe,4)),Qe(oe,5,Me(fe,5)),R(fe,oe),ne=Kt(ne),je.call(we,Te,ne);if(de.M)for(Pe=0;Pe<de.M.length;++Pe){if(we=K.data,je=we.addClassification,Te=ie,ne=de.M[Pe],fe=new W,pe(fe,2,ne.Y),ne.index&&pe(fe,1,ne.index),ne.label&&pe(fe,3,ne.label),ne.displayName&&pe(fe,4,ne.displayName),oe=ne=new qt,g=Me(fe,1),g!=null&&g!=null)if(nt(oe.g,8),he=oe.g,0<=g)nt(he,g);else{for(ge=0;9>ge;ge++)he.push(g&127|128),g>>=7;he.push(1)}Qe(oe,2,Me(fe,2)),he=Me(fe,3),he!=null&&(he=We(he),nt(oe.g,26),nt(oe.g,he.length),gt(oe,oe.g.end()),gt(oe,he)),he=Me(fe,4),he!=null&&(he=We(he),nt(oe.g,34),nt(oe.g,he.length),gt(oe,oe.g.end()),gt(oe,he)),R(fe,oe),ne=Kt(ne),je.call(we,Te,ne)}}K=K.data;break e;default:K={}}}switch(Z=K,ve=z.stream,z.type){case"video":C.pushTexture2d(Object.assign(Object.assign({},Z),{stream:ve,timestamp:M}));break;case"detections":se=Z,se.stream=ve,se.timestamp=M,C.pushDetectionList(se);break;default:throw Error("Unknown input config type: '"+z.type+"'")}}return v.i.send(C),w(te,v.C,4);case 4:C.delete(),te.g=0}})})};function er(o,l,c){return U(o,function v(){var M,C,T,E,B,z,Z=this,ve,se,te,K,Y,ie,de,Pe;return D(v,function(we){switch(we.g){case 1:if(!c)return we.return(l);for(M={},C=0,T=p(Object.keys(c)),E=T.next();!E.done;E=T.next())B=E.value,z=c[B],typeof z!="string"&&z.type==="texture"&&l[z.stream]!==void 0&&++C;1<C&&(Z.G=!1),ve=p(Object.keys(c)),E=ve.next();case 2:if(E.done){we.g=4;break}if(se=E.value,te=c[se],typeof te=="string")return de=M,Pe=se,w(we,jt(Z,se,l[te]),14);if(K=l[te.stream],te.type==="detection_list"){if(K){for(var je=K.getRectList(),Te=K.getLandmarksList(),ne=K.getClassificationsList(),fe=[],he=0;he<je.size();++he){var oe=je.get(he);e:{var ge=new Lt;for(oe=new Mt(oe);mt(oe);)switch(oe.i){case 13:var g=Ne(oe.g);pe(ge,1,g);break;case 21:g=Ne(oe.g),pe(ge,2,g);break;case 29:g=Ne(oe.g),pe(ge,3,g);break;case 37:g=Ne(oe.g),pe(ge,4,g);break;case 45:g=Ne(oe.g),pe(ge,5,g);break;case 48:for(var x=oe.g,N=128,H=0,q=g=0;4>q&&128<=N;q++)N=x.h[x.g++],H|=(N&127)<<7*q;if(128<=N&&(N=x.h[x.g++],H|=(N&127)<<28,g|=(N&127)>>4),128<=N)for(q=0;5>q&&128<=N;q++)N=x.h[x.g++],g|=(N&127)<<7*q+3;128>N?(x=H>>>0,N=g>>>0,(g=N&2147483648)&&(x=~x+1>>>0,N=~N>>>0,x==0&&(N=N+1>>>0)),x=4294967296*N+(x>>>0),g=g?-x:x):(x.l=!0,g=void 0),pe(ge,6,g);break;default:if(!L(ge,oe))break e}}ge={Z:Je(ge,1),$:Je(ge,2),height:Je(ge,3),width:Je(ge,4),rotation:Je(ge,5,0),X:kt(ge,6)},oe=ct(Te.get(he)),x=ne.get(he);e:for(g=new Se,x=new Mt(x);mt(x);)switch(x.i){case 10:g.addClassification(Dt(x,new W,Ie));break;default:if(!L(g,x))break e}ge={T:ge,O:oe,M:wn(g)},fe.push(ge)}je=fe}else je=[];M[se]=je,we.g=7;break}if(te.type==="proto_list"){if(K){for(je=Array(K.size()),Te=0;Te<K.size();Te++)je[Te]=K.get(Te);K.delete()}else je=[];M[se]=je,we.g=7;break}if(K===void 0){we.g=3;break}if(te.type==="float_list"){M[se]=K,we.g=7;break}if(te.type==="proto"){M[se]=K,we.g=7;break}if(te.type!=="texture")throw Error("Unknown output config type: '"+te.type+"'");return Y=Z.s[se],Y||(Y=new Rt(Z.h,Z.D),Z.s[se]=Y),w(we,Pn(Y,K,Z.G),13);case 13:ie=we.h,M[se]=ie;case 7:te.transform&&M[se]&&(M[se]=te.transform(M[se])),we.g=3;break;case 14:de[Pe]=we.h;case 3:E=ve.next(),we.g=2;break;case 4:return we.return(M)}})})}function jt(o,l,c){return U(o,function v(){var M=this,C;return D(v,function(T){return typeof c=="number"||c instanceof Uint8Array||c instanceof M.h.Uint8BlobList?T.return(c):c instanceof M.h.Texture2dDataOut?(C=M.s[l],C||(C=new Rt(M.h,M.D),M.s[l]=C),T.return(Pn(C,c,M.G))):T.return(void 0)})})}function Sn(o,l){for(var c=l.name||"$",v=[].concat(h(l.wants)),M=new o.h.StringList,C=p(l.wants),T=C.next();!T.done;T=C.next())M.push_back(T.value);C=o.h.PacketListener.implement({onResults:function(E){for(var B={},z=0;z<l.wants.length;++z)B[v[z]]=E.get(z);var Z=o.listeners[c];Z&&(o.C=er(o,B,l.outs).then(function(ve){ve=Z(ve);for(var se=0;se<l.wants.length;++se){var te=B[v[se]];typeof te=="object"&&te.hasOwnProperty&&te.hasOwnProperty("delete")&&te.delete()}ve&&(o.C=ve)}))}}),o.i.attachMultiListener(M,C),M.delete()}a.onResults=function(o,l){this.listeners[l||"$"]=o},$("Solution",In),$("OptionType",{BOOL:0,NUMBER:1,aa:2,0:"BOOL",1:"NUMBER",2:"STRING"});function Bn(o){switch(o===void 0&&(o=0),o){case 1:return"pose_landmark_full.tflite";case 2:return"pose_landmark_heavy.tflite";default:return"pose_landmark_lite.tflite"}}function jn(o){var l=this;o=o||{},this.g=new In({locateFile:o.locateFile,files:function(c){return[{url:"pose_solution_packed_assets_loader.js"},{simd:!1,url:"pose_solution_wasm_bin.js"},{simd:!0,url:"pose_solution_simd_wasm_bin.js"},{data:!0,url:Bn(c.modelComplexity)}]},graph:{url:"pose_web.binarypb"},listeners:[{wants:["pose_landmarks","world_landmarks","segmentation_mask","image_transformed"],outs:{image:{type:"texture",stream:"image_transformed"},poseLandmarks:{type:"proto",stream:"pose_landmarks",transform:ct},poseWorldLandmarks:{type:"proto",stream:"world_landmarks",transform:ct},segmentationMask:{type:"texture",stream:"segmentation_mask"}}}],inputs:{image:{type:"video",stream:"input_frames_gpu"}},options:{useCpuInference:{type:0,graphOptionXref:{calculatorType:"InferenceCalculator",fieldName:"use_cpu_inference"},default:"iPad Simulator;iPhone Simulator;iPod Simulator;iPad;iPhone;iPod".split(";").includes(navigator.platform)||navigator.userAgent.includes("Mac")&&"ontouchend"in document},selfieMode:{type:0,graphOptionXref:{calculatorType:"GlScalerCalculator",calculatorIndex:1,fieldName:"flip_horizontal"}},modelComplexity:{type:1,graphOptionXref:{calculatorType:"ConstantSidePacketCalculator",calculatorName:"ConstantSidePacketCalculatorModelComplexity",fieldName:"int_value"},onChange:function(c){return U(l,function v(){var M,C,T=this,E;return D(v,function(B){return B.g==1?(M=Bn(c),C="third_party/mediapipe/modules/pose_landmark/"+M,w(B,dn(T.g,M),2)):(E=B.h,T.g.overrideFile(C,E),B.return(!0))})})}},smoothLandmarks:{type:0,graphOptionXref:{calculatorType:"ConstantSidePacketCalculator",calculatorName:"ConstantSidePacketCalculatorSmoothLandmarks",fieldName:"bool_value"}},enableSegmentation:{type:0,graphOptionXref:{calculatorType:"ConstantSidePacketCalculator",calculatorName:"ConstantSidePacketCalculatorEnableSegmentation",fieldName:"bool_value"}},smoothSegmentation:{type:0,graphOptionXref:{calculatorType:"ConstantSidePacketCalculator",calculatorName:"ConstantSidePacketCalculatorSmoothSegmentation",fieldName:"bool_value"}},minDetectionConfidence:{type:1,graphOptionXref:{calculatorType:"TensorsToDetectionsCalculator",calculatorName:"poselandmarkgpu__posedetectiongpu__TensorsToDetectionsCalculator",fieldName:"min_score_thresh"}},minTrackingConfidence:{type:1,graphOptionXref:{calculatorType:"ThresholdingCalculator",calculatorName:"poselandmarkgpu__poselandmarkbyroigpu__tensorstoposelandmarksandsegmentation__ThresholdingCalculator",fieldName:"threshold"}}}})}a=jn.prototype,a.reset=function(){this.g.reset()},a.close=function(){return this.g.close(),Promise.resolve()},a.onResults=function(o){this.g.onResults(o)},a.initialize=function(){return U(this,function o(){var l=this;return D(o,function(c){return w(c,l.g.initialize(),0)})})},a.send=function(o,l){return U(this,function c(){var v=this;return D(c,function(M){return w(M,v.g.send(o,l),0)})})},a.setOptions=function(o){this.g.setOptions(o)},$("Pose",jn),$("POSE_CONNECTIONS",[[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32]]),$("POSE_LANDMARKS",{NOSE:0,LEFT_EYE_INNER:1,LEFT_EYE:2,LEFT_EYE_OUTER:3,RIGHT_EYE_INNER:4,RIGHT_EYE:5,RIGHT_EYE_OUTER:6,LEFT_EAR:7,RIGHT_EAR:8,LEFT_RIGHT:9,RIGHT_LEFT:10,LEFT_SHOULDER:11,RIGHT_SHOULDER:12,LEFT_ELBOW:13,RIGHT_ELBOW:14,LEFT_WRIST:15,RIGHT_WRIST:16,LEFT_PINKY:17,RIGHT_PINKY:18,LEFT_INDEX:19,RIGHT_INDEX:20,LEFT_THUMB:21,RIGHT_THUMB:22,LEFT_HIP:23,RIGHT_HIP:24,LEFT_KNEE:25,RIGHT_KNEE:26,LEFT_ANKLE:27,RIGHT_ANKLE:28,LEFT_HEEL:29,RIGHT_HEEL:30,LEFT_FOOT_INDEX:31,RIGHT_FOOT_INDEX:32}),$("POSE_LANDMARKS_LEFT",{LEFT_EYE_INNER:1,LEFT_EYE:2,LEFT_EYE_OUTER:3,LEFT_EAR:7,LEFT_RIGHT:9,LEFT_SHOULDER:11,LEFT_ELBOW:13,LEFT_WRIST:15,LEFT_PINKY:17,LEFT_INDEX:19,LEFT_THUMB:21,LEFT_HIP:23,LEFT_KNEE:25,LEFT_ANKLE:27,LEFT_HEEL:29,LEFT_FOOT_INDEX:31}),$("POSE_LANDMARKS_RIGHT",{RIGHT_EYE_INNER:4,RIGHT_EYE:5,RIGHT_EYE_OUTER:6,RIGHT_EAR:8,RIGHT_LEFT:10,RIGHT_SHOULDER:12,RIGHT_ELBOW:14,RIGHT_WRIST:16,RIGHT_PINKY:18,RIGHT_INDEX:20,RIGHT_THUMB:22,RIGHT_HIP:24,RIGHT_KNEE:26,RIGHT_ANKLE:28,RIGHT_HEEL:30,RIGHT_FOOT_INDEX:32}),$("POSE_LANDMARKS_NEUTRAL",{NOSE:0}),$("VERSION","0.4.1633558788")}).call(An);/**
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
    */var Si=function(a,e){return(Si=Object.setPrototypeOf||{__proto__:[]}instanceof Array&&function(r,i){r.__proto__=i}||function(r,i){for(var s in i)i.hasOwnProperty(s)&&(r[s]=i[s])})(a,e)};function xi(a,e){function r(){this.constructor=a}Si(a,e),a.prototype=e===null?Object.create(e):(r.prototype=e.prototype,new r)}var xe=function(){return(xe=Object.assign||function(a){for(var e,r=1,i=arguments.length;r<i;r++)for(var s in e=arguments[r])Object.prototype.hasOwnProperty.call(e,s)&&(a[s]=e[s]);return a}).apply(this,arguments)};function be(a,e,r,i){return new(r||(r=Promise))(function(s,t){function u(d){try{h(i.next(d))}catch(n){t(n)}}function p(d){try{h(i.throw(d))}catch(n){t(n)}}function h(d){var n;d.done?s(d.value):(n=d.value,n instanceof r?n:new r(function(f){f(n)})).then(u,p)}h((i=i.apply(a,e||[])).next())})}function _e(a,e){var r,i,s,t,u={label:0,sent:function(){if(1&s[0])throw s[1];return s[1]},trys:[],ops:[]};return t={next:p(0),throw:p(1),return:p(2)},typeof Symbol=="function"&&(t[Symbol.iterator]=function(){return this}),t;function p(h){return function(d){return function(n){if(r)throw new TypeError("Generator is already executing.");for(;u;)try{if(r=1,i&&(s=2&n[0]?i.return:n[0]?i.throw||((s=i.return)&&s.call(i),0):i.next)&&!(s=s.call(i,n[1])).done)return s;switch(i=0,s&&(n=[2&n[0],s.value]),n[0]){case 0:case 1:s=n;break;case 4:return u.label++,{value:n[1],done:!1};case 5:u.label++,i=n[1],n=[0];continue;case 7:n=u.ops.pop(),u.trys.pop();continue;default:if(s=u.trys,!((s=s.length>0&&s[s.length-1])||n[0]!==6&&n[0]!==2)){u=0;continue}if(n[0]===3&&(!s||n[1]>s[0]&&n[1]<s[3])){u.label=n[1];break}if(n[0]===6&&u.label<s[1]){u.label=s[1],s=n;break}if(s&&u.label<s[2]){u.label=s[2],u.ops.push(n);break}s[2]&&u.ops.pop(),u.trys.pop();continue}n=e.call(a,u)}catch(f){n=[6,f],i=0}finally{r=s=0}if(5&n[0])throw n[1];return{value:n[0]?n[1]:void 0,done:!0}}([h,d])}}}function nn(){for(var a=0,e=0,r=arguments.length;e<r;e++)a+=arguments[e].length;var i=Array(a),s=0;for(e=0;e<r;e++)for(var t=arguments[e],u=0,p=t.length;u<p;u++,s++)i[s]=t[u];return i}var St=["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"],En=["nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer","left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index"];function Gn(a){return a instanceof SVGAnimatedLength?a.baseVal.value:a}function Mi(a){return be(this,void 0,void 0,function(){var e,r;return _e(this,function(i){switch(i.label){case 0:return e=document.createElement("canvas"),a instanceof bn?[4,Ar(a,e)]:[3,2];case 1:return i.sent(),[3,3];case 2:e.width=Gn(a.width),e.height=Gn(a.height),r=e.getContext("2d"),a instanceof ImageData?r.putImageData(a,0,0):r.drawImage(a,0,0),i.label=3;case 3:return[2,e]}})})}function Ai(a){return be(this,void 0,void 0,function(){var e,r,i,s,t,u;return _e(this,function(p){switch(p.label){case 0:return a instanceof bn?(e=a.shape.slice(0,2),r=e[0],i=e[1],s=ImageData.bind,[4,Ar(a)]):[3,2];case 1:return[2,new(s.apply(ImageData,[void 0,p.sent(),i,r]))];case 2:return t=document.createElement("canvas"),u=t.getContext("2d"),t.width=Gn(a.width),t.height=Gn(a.height),u.drawImage(a,0,0),[2,u.getImageData(0,0,t.width,t.height)]}})})}function Fh(a){return be(this,void 0,void 0,function(){var e,r;return _e(this,function(i){switch(i.label){case 0:return a instanceof SVGImageElement||a instanceof OffscreenCanvas?[4,Mi(a)]:[3,2];case 1:return r=i.sent(),[3,3];case 2:r=a,i.label=3;case 3:return e=r,[2,hs(e,4)]}})})}function Ti(a){if(a<0||a>=256)throw new Error("Mask value must be in range [0, 255] but got "+a);if(!Number.isInteger(a))throw new Error("Mask value must be an integer but got "+a)}var xn={runtime:"mediapipe",enableSmoothing:!0,enableSegmentation:!1,smoothSegmentation:!0,modelType:"full"},Eh=function(){function a(e){this.mask=e}return a.prototype.toCanvasImageSource=function(){return be(this,void 0,void 0,function(){return _e(this,function(e){return[2,this.mask]})})},a.prototype.toImageData=function(){return be(this,void 0,void 0,function(){return _e(this,function(e){return[2,Ai(this.mask)]})})},a.prototype.toTensor=function(){return be(this,void 0,void 0,function(){return _e(this,function(e){return[2,Fh(this.mask)]})})},a.prototype.getUnderlyingType=function(){return"canvasimagesource"},a}();function Nh(a){return Ti(a),"person"}var Ch=function(){function a(e){var r,i=this;switch(this.width=0,this.height=0,this.selfieMode=!1,this.poseSolution=new Rh.Pose({locateFile:function(s,t){return e.solutionPath?e.solutionPath.replace(/\/+$/,"")+"/"+s:t+"/"+s}}),e.modelType){case"lite":r=0;break;case"heavy":r=2;break;case"full":default:r=1}this.poseSolution.setOptions({modelComplexity:r,smoothLandmarks:e.enableSmoothing,enableSegmentation:e.enableSegmentation,smoothSegmentation:e.smoothSegmentation,selfieMode:this.selfieMode}),this.poseSolution.onResults(function(s){if(i.height=s.image.height,i.width=s.image.width,s.poseLandmarks==null)i.poses=[];else{var t=i.translateOutput(s.poseLandmarks,s.poseWorldLandmarks);s.segmentationMask&&(t.segmentation={maskValueToLabel:Nh,mask:new Eh(s.segmentationMask)}),i.poses=[t]}})}return a.prototype.translateOutput=function(e,r){var i=this,s={keypoints:e.map(function(t,u){return{x:t.x*i.width,y:t.y*i.height,z:t.z,score:t.visibility,name:En[u]}})};return r!=null&&(s.keypoints3D=r.map(function(t,u){return{x:t.x,y:t.y,z:t.z,score:t.visibility,name:En[u]}})),s},a.prototype.estimatePoses=function(e,r,i){return be(this,void 0,void 0,function(){var s,t;return _e(this,function(u){switch(u.label){case 0:return r&&r.flipHorizontal&&r.flipHorizontal!==this.selfieMode&&(this.selfieMode=r.flipHorizontal,this.poseSolution.setOptions({selfieMode:this.selfieMode})),e instanceof bn?(t=ImageData.bind,[4,Ar(e)]):[3,2];case 1:return s=new(t.apply(ImageData,[void 0,u.sent(),e.shape[1],e.shape[0]])),[3,3];case 2:s=e,u.label=3;case 3:return e=s,[4,this.poseSolution.send({image:e},i)];case 4:return u.sent(),[2,this.poses]}})})},a.prototype.dispose=function(){this.poseSolution.close()},a.prototype.reset=function(){this.poseSolution.reset()},a.prototype.initialize=function(){return this.poseSolution.initialize()},a}();function Oh(a){return be(this,void 0,void 0,function(){var e,r;return _e(this,function(i){switch(i.label){case 0:return e=function(s){if(s==null)return xe({},xn);var t=xe({},s);return t.runtime="mediapipe",t.enableSegmentation==null&&(t.enableSegmentation=xn.enableSegmentation),t.enableSmoothing==null&&(t.enableSmoothing=xn.enableSmoothing),t.smoothSegmentation==null&&(t.smoothSegmentation=xn.smoothSegmentation),t.modelType==null&&(t.modelType=xn.modelType),t}(a),[4,(r=new Ch(e)).initialize()];case 1:return i.sent(),[2,r]}})})}function Nn(a){return a instanceof bn?{height:a.shape[0],width:a.shape[1]}:{height:a.height,width:a.width}}function Ri(a){return a-2*Math.PI*Math.floor((a+Math.PI)/(2*Math.PI))}function Fr(a){return a instanceof bn?a:hs(a)}function Fi(a,e,r){return wr(r,"inputResolution"),[1/r.width*a[0][0]*e.width,1/r.height*a[0][1]*e.width,a[0][3]*e.width,1/r.width*a[1][0]*e.height,1/r.height*a[1][1]*e.height,a[1][3]*e.height,0,0]}function wr(a,e){et(a.width!==0,function(){return e+" width cannot be 0."}),et(a.height!==0,function(){return e+" height cannot be 0."})}function ir(a,e,r){var i=r.rotationVectorStartKeypointIndex,s=r.rotationVectorEndKeypointIndex,t=a.locationData,u=t.relativeKeypoints[i].x*e.width,p=t.relativeKeypoints[i].y*e.height,h=t.relativeKeypoints[s].x*e.width,d=t.relativeKeypoints[s].y*e.height,n=2*Math.sqrt((h-u)*(h-u)+(d-p)*(d-p)),f=function(_,m,y){var b,k=_.locationData,S=y.rotationVectorStartKeypointIndex,I=y.rotationVectorEndKeypointIndex;b=y.rotationVectorTargetAngle?y.rotationVectorTargetAngle:Math.PI*y.rotationVectorTargetAngleDegree/180;var w=k.relativeKeypoints[S].x*m.width,A=k.relativeKeypoints[S].y*m.height,O=k.relativeKeypoints[I].x*m.width,F=k.relativeKeypoints[I].y*m.height;return Ri(b-Math.atan2(-(F-A),O-w))}(a,e,r);return{xCenter:u/e.width,yCenter:p/e.height,width:n/e.width,height:n/e.height,rotation:f}}function Ei(a){if(a.length!==16)throw new Error("Array length must be 16 but got "+a.length);return[[a[0],a[1],a[2],a[3]],[a[4],a[5],a[6],a[7]],[a[8],a[9],a[10],a[11]],[a[12],a[13],a[14],a[15]]]}function or(a,e,r,i,s,t,u){return a[e][s]*(a[r][t]*a[i][u]-a[r][u]*a[i][t])}function Xe(a,e,r){var i=(e+1)%4,s=(e+2)%4,t=(e+3)%4,u=(r+1)%4,p=(r+2)%4,h=(r+3)%4;return or(a,i,s,t,u,p,h)+or(a,s,t,i,u,p,h)+or(a,t,i,s,u,p,h)}function Wr(a,e,r){r===void 0&&(r={ignoreRotation:!1});for(var i=[],s=0,t=a;s<t.length;s++){var u=t[s],p=u.x-.5,h=u.y-.5,d=r.ignoreRotation?0:e.rotation,n=Math.cos(d)*p-Math.sin(d)*h,f=Math.sin(d)*p+Math.cos(d)*h;n=n*e.width+e.xCenter,f=f*e.height+e.yCenter;var _=u.z*e.width,m=xe({},u);m.x=n,m.y=f,m.z=_,i.push(m)}return i}function Ni(a,e){var r=function(i,s,t,u){var p=s-i,h=u-t;if(p===0)throw new Error("Original min and max are both "+i+", range cannot be 0.");var d=h/p;return{scale:d,offset:t-i*d}}(0,255,e[0],e[1]);return Ye(function(){return ot($e(a,r.scale),r.offset)})}function kr(a,e,r){var i,s,t,u,p,h,d,n,f,_,m,y,b,k,S=e.outputTensorSize,I=e.keepAspectRatio,w=e.borderMode,A=e.outputTensorFloatRange,O=Nn(a),F=function(D,j){return j?{xCenter:j.xCenter*D.width,yCenter:j.yCenter*D.height,width:j.width*D.width,height:j.height*D.height,rotation:j.rotation}:{xCenter:.5*D.width,yCenter:.5*D.height,width:D.width,height:D.height,rotation:0}}(O,r),P=function(D,j,G){if(G===void 0&&(G=!1),!G)return{top:0,left:0,right:0,bottom:0};var X=j.height,$=j.width;wr(j,"targetSize"),wr(D,"roi");var le,Q,ce=X/$,ke=D.height/D.width,me=0,We=0;return ce>ke?(le=D.width,Q=D.width*ce,We=(1-ke/ce)/2):(le=D.height/ce,Q=D.height,me=(1-ce/ke)/2),D.width=le,D.height=Q,{top:We,left:me,right:me,bottom:We}}(F,S,I),V=(i=F,s=O.width,t=O.height,u=!1,p=i.width,h=i.height,d=u?-1:1,n=Math.cos(i.rotation),f=Math.sin(i.rotation),_=i.xCenter,m=i.yCenter,y=1/s,b=1/t,(k=new Array(16))[0]=p*n*d*y,k[1]=-h*f*y,k[2]=0,k[3]=(-.5*p*n*d+.5*h*f+_)*y,k[4]=p*f*d*b,k[5]=h*n*b,k[6]=0,k[7]=(-.5*h*n-.5*p*f*d+m)*b,k[8]=0,k[9]=0,k[10]=p*y,k[11]=0,k[12]=0,k[13]=0,k[14]=0,k[15]=1,Ei(k));return{imageTensor:Ye(function(){var D=Fr(a),j=rn(Fi(V,O,S),[1,8]),G=w==="zero"?"constant":"nearest",X=an.transform(Tn(Rn(D,"float32")),j,"bilinear",G,0,[S.height,S.width]);return A!=null?Ni(X,A):X}),padding:P,transformationMatrix:V}}function Hr(a,e,r,i){return i===1?.5*(a+e):a+(e-a)*r/(i-1)}function Ph(a){return Ye(function(){var e=function(s){return Ye(function(){return[st(s,[0,0,0],[1,-1,1]),st(s,[0,0,1],[1,-1,-1])]})}(a),r=e[0],i=e[1];return{boxes:Re(i),logits:Re(r)}})}function Ci(a){return a!=null&&a.currentTime!=null}function Ur(a){for(var e={locationData:{relativeKeypoints:[]}},r=Number.MAX_SAFE_INTEGER,i=Number.MIN_SAFE_INTEGER,s=Number.MAX_SAFE_INTEGER,t=Number.MIN_SAFE_INTEGER,u=0;u<a.length;++u){var p=a[u];r=Math.min(r,p.x),i=Math.max(i,p.x),s=Math.min(s,p.y),t=Math.max(t,p.y),e.locationData.relativeKeypoints.push({x:p.x,y:p.y})}return e.locationData.relativeBoundingBox={xMin:r,yMin:s,xMax:i,yMax:t,width:i-r,height:t-s},e}function Dh(a,e,r,i){return be(this,void 0,void 0,function(){var s,t,u,p,h;return _e(this,function(d){switch(d.label){case 0:return a.sort(function(n,f){return Math.max.apply(Math,f.score)-Math.max.apply(Math,n.score)}),s=rn(a.map(function(n){return[n.locationData.relativeBoundingBox.yMin,n.locationData.relativeBoundingBox.xMin,n.locationData.relativeBoundingBox.yMax,n.locationData.relativeBoundingBox.xMax]})),t=Mn(a.map(function(n){return n.score[0]})),[4,an.nonMaxSuppressionAsync(s,t,e,r)];case 1:return[4,(u=d.sent()).array()];case 2:return p=d.sent(),h=a.filter(function(n,f){return p.indexOf(f)>-1}),Ze([s,t,u]),[2,h]}})})}function Oi(a,e){return a.map(function(r){var i=xe(xe({},r),{x:r.x*e.width,y:r.y*e.height});return r.z!=null&&(i.z=r.z*e.width),i})}function Lh(a,e,r){return be(this,void 0,void 0,function(){var i,s,t,u,p,h,d,n,f,_,m,y,b,k,S,I,w,A,O,F,P,V,D,j;return _e(this,function(G){switch(G.label){case 0:if(i=Re(e,[0]),s=i.shape,t=s[0],u=s[1],p=s[2],a.length!==p)throw new Error("Expected heatmap to have same number of channels as the number of landmarks. But got landmarks length: "+a.length+", heatmap length: "+p);return h=[],[4,i.buffer()];case 1:for(d=G.sent(),n=0;n<a.length;n++)if(f=a[n],_=xe({},f),h.push(_),m=Math.trunc(_.x*u),y=Math.trunc(_.y*t),!(m<0||m>=u||y<0||m>=t)){for(b=Math.trunc((r.kernelSize-1)/2),k=Math.max(0,m-b),S=Math.min(u,m+b+1),I=Math.max(0,y-b),w=Math.min(t,y+b+1),A=0,O=0,F=0,P=0,V=I;V<w;++V)for(D=k;D<S;++D)j=d.get(V,D,n),A+=j,P=Math.max(P,j),O+=D*j,F+=V*j;P>=r.minConfidenceToRefine&&A>0&&(_.x=O/u/A,_.y=F/t/A)}return i.dispose(),[2,h]}})})}function zr(a,e){var r=e.left,i=e.top,s=e.left+e.right,t=e.top+e.bottom;return a.map(function(u){return xe(xe({},u),{x:(u.x-r)/(1-s),y:(u.y-i)/(1-t),z:u.z/(1-s)})})}function Bh(a,e,r){return Un()==="webgl"?function(i,s,t){var u=t.combineWithPreviousRatio.toFixed(2),p={variableNames:["prevMask","newMask"],outputShape:i.shape,userCode:`
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
`},h=ou();return Ye(function(){var d=h.compileAndRun(p,[i,s]);return ds().makeTensorFromDataId(d.dataId,d.shape,d.dtype)})}(a,e,r):Ye(function(){var i=mn(e,.5),s=uu(i),t=mn(1,lu(1,$e(s,ot(5.68842,$e(s,ot(-.748699,$e(s,ot(-57.8051,$e(s,ot(291.309,$e(s,-624.717)))))))))));return ot(e,$e(mn(a,e),$e(t,r.combineWithPreviousRatio)))})}function jh(a,e,r){return be(this,void 0,void 0,function(){var i,s,t,u,p;return _e(this,function(h){switch(h.label){case 0:return i=a[0],s=a[1],t=function(d,n,f){return Ye(function(){var _,m,y,b;f.reverseOutputOrder?(m=Re(st(d,[0,f.boxCoordOffset+0],[-1,1])),_=Re(st(d,[0,f.boxCoordOffset+1],[-1,1])),b=Re(st(d,[0,f.boxCoordOffset+2],[-1,1])),y=Re(st(d,[0,f.boxCoordOffset+3],[-1,1]))):(_=Re(st(d,[0,f.boxCoordOffset+0],[-1,1])),m=Re(st(d,[0,f.boxCoordOffset+1],[-1,1])),y=Re(st(d,[0,f.boxCoordOffset+2],[-1,1])),b=Re(st(d,[0,f.boxCoordOffset+3],[-1,1]))),m=ot($e(it(m,f.xScale),n.w),n.x),_=ot($e(it(_,f.yScale),n.h),n.y),f.applyExponentialOnBoxSize?(y=$e(Pr(it(y,f.hScale)),n.h),b=$e(Pr(it(b,f.wScale)),n.w)):(y=$e(it(y,f.hScale),n.h),b=$e(it(b,f.wScale),n.h));var k=mn(_,it(y,2)),S=mn(m,it(b,2)),I=ot(_,it(y,2)),w=ot(m,it(b,2)),A=gr([Vt(k,[f.numBoxes,1]),Vt(S,[f.numBoxes,1]),Vt(I,[f.numBoxes,1]),Vt(w,[f.numBoxes,1])],1);if(f.numKeypoints)for(var O=0;O<f.numKeypoints;++O){var F=f.keypointCoordOffset+O*f.numValuesPerKeypoint,P=void 0,V=void 0;f.reverseOutputOrder?(P=Re(st(d,[0,F],[-1,1])),V=Re(st(d,[0,F+1],[-1,1]))):(V=Re(st(d,[0,F],[-1,1])),P=Re(st(d,[0,F+1],[-1,1])));var D=ot($e(it(P,f.xScale),n.w),n.x),j=ot($e(it(V,f.yScale),n.h),n.y);A=gr([A,Vt(D,[f.numBoxes,1]),Vt(j,[f.numBoxes,1])],1)}return A})}(s,e,r),u=Ye(function(){var d=i;return r.sigmoidScore?(r.scoreClippingThresh!=null&&(d=cu(i,-r.scoreClippingThresh,r.scoreClippingThresh)),d=Tr(d)):d}),[4,Vh(t,u,r)];case 1:return p=h.sent(),Ze([t,u]),[2,p]}})})}function Vh(a,e,r){return be(this,void 0,void 0,function(){var i,s,t,u,p,h,d,n,f,_,m,y;return _e(this,function(b){switch(b.label){case 0:return i=[],[4,a.data()];case 1:return s=b.sent(),[4,e.data()];case 2:for(t=b.sent(),u=0;u<r.numBoxes;++u)if(!(r.minScoreThresh!=null&&t[u]<r.minScoreThresh||(p=u*r.numCoords,h=Wh(s[p+0],s[p+1],s[p+2],s[p+3],t[u],r.flipVertically,u),(d=h.locationData.relativeBoundingBox).width<0||d.height<0))){if(r.numKeypoints>0)for((n=h.locationData).relativeKeypoints=[],f=r.numKeypoints*r.numValuesPerKeypoint,_=0;_<f;_+=r.numValuesPerKeypoint)m=p+r.keypointCoordOffset+_,y={x:s[m+0],y:r.flipVertically?1-s[m+1]:s[m+1]},n.relativeKeypoints.push(y);i.push(h)}return[2,i]}})})}function Wh(a,e,r,i,s,t,u){return{score:[s],ind:u,locationData:{relativeBoundingBox:{xMin:e,yMin:t?1-r:a,xMax:i,yMax:t?1-a:r,width:i-e,height:r-a}}}}function Hh(a,e){return a==="none"?e:function(r){return 1/(1+Math.exp(-r))}(e)}function $r(a,e,r,i){return be(this,void 0,void 0,function(){var s,t,u,p,h,d,n,f;return _e(this,function(_){switch(_.label){case 0:return r=r||e.flipHorizontally||!1,i=i||e.flipVertically||!1,s=a.size,t=s/e.numLandmarks,[4,a.data()];case 1:for(u=_.sent(),p=[],h=0;h<e.numLandmarks;++h)d=h*t,(f={x:0,y:0}).x=r?e.inputImageWidth-u[d]:u[d],t>1&&(f.y=i?e.inputImageHeight-u[d+1]:u[d+1]),t>2&&(f.z=u[d+2]),t>3&&(f.score=Hh(e.visibilityActivation,u[d+3])),p.push(f);for(n=0;n<p.length;++n)(f=p[n]).x=f.x/e.inputImageWidth,f.y=f.y/e.inputImageHeight,f.z=f.z/e.inputImageWidth/(e.normalizeZ||1);return[2,p]}})})}function Gr(a,e,r){var i=a.width,s=a.height,t=a.rotation;if(r.rotation==null&&r.rotationDegree==null||(t=function(n,f){return f.rotation!=null?n+=f.rotation:f.rotationDegree!=null&&(n+=Math.PI*f.rotationDegree/180),Ri(n)}(t,r)),t===0)a.xCenter=a.xCenter+i*r.shiftX,a.yCenter=a.yCenter+s*r.shiftY;else{var u=(e.width*i*r.shiftX*Math.cos(t)-e.height*s*r.shiftY*Math.sin(t))/e.width,p=(e.width*i*r.shiftX*Math.sin(t)+e.height*s*r.shiftY*Math.cos(t))/e.height;a.xCenter=a.xCenter+u,a.yCenter=a.yCenter+p}if(r.squareLong){var h=Math.max(i*e.width,s*e.height);i=h/e.width,s=h/e.height}else if(r.squareShort){var d=Math.min(i*e.width,s*e.height);i=d/e.width,s=d/e.height}return a.width=i*r.scaleX,a.height=s*r.scaleY,a}function Uh(a,e){return a.map(function(r){var i=xe(xe({},r),{x:r.x/e.width,y:r.y/e.height});return r.z!=null&&(r.z=r.z/e.width),i})}var Wt=function(){function a(e){this.alpha=e,this.initialized=!1}return a.prototype.apply=function(e,r){var i;return this.initialized?i=r==null?this.storedValue+this.alpha*(e-this.storedValue):this.storedValue+this.alpha*r*Math.asinh((e-this.storedValue)/r):(i=e,this.initialized=!0),this.rawValue=e,this.storedValue=i,i},a.prototype.applyWithAlpha=function(e,r,i){return this.alpha=r,this.apply(e,i)},a.prototype.hasLastRawValue=function(){return this.initialized},a.prototype.lastRawValue=function(){return this.rawValue},a.prototype.reset=function(){this.initialized=!1},a}(),ur=function(){function a(e){this.frequency=e.frequency,this.minCutOff=e.minCutOff,this.beta=e.beta,this.thresholdCutOff=e.thresholdCutOff,this.thresholdBeta=e.thresholdBeta,this.derivateCutOff=e.derivateCutOff,this.x=new Wt(this.getAlpha(this.minCutOff)),this.dx=new Wt(this.getAlpha(this.derivateCutOff)),this.lastTimestamp=0}return a.prototype.apply=function(e,r,i){if(e==null)return e;var s=Math.trunc(r);if(this.lastTimestamp>=s)return e;this.lastTimestamp!==0&&s!==0&&(this.frequency=1/(1e-6*(s-this.lastTimestamp))),this.lastTimestamp=s;var t=this.x.hasLastRawValue()?(e-this.x.lastRawValue())*i*this.frequency:0,u=this.dx.applyWithAlpha(t,this.getAlpha(this.derivateCutOff)),p=this.minCutOff+this.beta*Math.abs(u),h=this.thresholdCutOff!=null?this.thresholdCutOff+this.thresholdBeta*Math.abs(u):null;return this.x.applyWithAlpha(e,this.getAlpha(p),h)},a.prototype.getAlpha=function(e){return 1/(1+this.frequency/(2*Math.PI*e))},a}(),Ir=function(){function a(e){this.config=e}return a.prototype.apply=function(e,r,i){var s=this;if(e==null)return this.reset(),null;this.initializeFiltersIfEmpty(e);var t=1;if(!this.config.disableValueScaling){if(i<this.config.minAllowedObjectScale)return nn(e);t=1/i}return e.map(function(u,p){var h=xe(xe({},u),{x:s.xFilters[p].apply(u.x,r,t),y:s.yFilters[p].apply(u.y,r,t)});return u.z!=null&&(h.z=s.zFilters[p].apply(u.z,r,t)),h})},a.prototype.reset=function(){this.xFilters=null,this.yFilters=null,this.zFilters=null},a.prototype.initializeFiltersIfEmpty=function(e){var r=this;this.xFilters!=null&&this.xFilters.length===e.length||(this.xFilters=e.map(function(i){return new ur(r.config)}),this.yFilters=e.map(function(i){return new ur(r.config)}),this.zFilters=e.map(function(i){return new ur(r.config)}))},a}(),lr=function(){function a(e){this.config=e,this.window=[],this.lowPassFilter=new Wt(1),this.lastValue=0,this.lastValueScale=1,this.lastTimestamp=-1}return a.prototype.apply=function(e,r,i){if(e==null)return e;var s,t=Math.trunc(r);if(this.lastTimestamp>=t)return e;if(this.lastTimestamp===-1)s=1;else{for(var u=e*i-this.lastValue*this.lastValueScale,p=t-this.lastTimestamp,h=u,d=p,n=(1+this.window.length)*(1e6/30),f=0,_=this.window;f<_.length;f++){var m=_[f];if(d+m.duration>n)break;h+=m.distance,d+=m.duration}var y=h/(1e-6*d);s=1-1/(1+this.config.velocityScale*Math.abs(y)),this.window.unshift({distance:u,duration:p}),this.window.length>this.config.windowSize&&this.window.pop()}return this.lastValue=e,this.lastValueScale=i,this.lastTimestamp=t,this.lowPassFilter.applyWithAlpha(e,s)},a}(),zh=function(){function a(e){this.config=e}return a.prototype.apply=function(e,r,i){var s=this;if(e==null)return this.reset(),null;var t=1;if(!this.config.disableValueScaling){if(i<this.config.minAllowedObjectScale)return nn(e);t=1/i}return this.initializeFiltersIfEmpty(e),e.map(function(u,p){var h=xe(xe({},u),{x:s.xFilters[p].apply(u.x,r,t),y:s.yFilters[p].apply(u.y,r,t)});return u.z!=null&&(h.z=s.zFilters[p].apply(u.z,r,t)),h})},a.prototype.reset=function(){this.xFilters=null,this.yFilters=null,this.zFilters=null},a.prototype.initializeFiltersIfEmpty=function(e){var r=this;this.xFilters!=null&&this.xFilters.length===e.length||(this.xFilters=e.map(function(i){return new lr(r.config)}),this.yFilters=e.map(function(i){return new lr(r.config)}),this.zFilters=e.map(function(i){return new lr(r.config)}))},a}(),cr=function(){function a(e){if(e.velocityFilter!=null)this.keypointsFilter=new zh(e.velocityFilter);else{if(e.oneEuroFilter==null)throw new Error("Either configure velocityFilter or oneEuroFilter, but got "+e+".");this.keypointsFilter=new Ir(e.oneEuroFilter)}}return a.prototype.apply=function(e,r,i,s,t){if(s===void 0&&(s=!1),e==null)return this.keypointsFilter.reset(),null;var u=t!=null?function(d,n){return(d.width*n.width+d.height*n.height)/2}(t,i):1,p=s?Oi(e,i):e,h=this.keypointsFilter.apply(p,r,u);return s?Uh(h,i):h},a}(),qr=function(){function a(e){this.alpha=e.alpha}return a.prototype.apply=function(e){var r=this;if(e==null)return this.visibilityFilters=null,null;this.visibilityFilters!=null&&this.visibilityFilters.length===e.length||(this.visibilityFilters=e.map(function(p){return new Wt(r.alpha)}));for(var i=[],s=0;s<e.length;++s){var t=e[s],u=xe({},t);u.score=this.visibilityFilters[s].apply(t.score),i.push(u)}return i},a}(),$h={reduceBoxesInLowestlayer:!1,interpolatedScaleAspectRatio:1,featureMapHeight:[],featureMapWidth:[],numLayers:5,minScale:.1484375,maxScale:.75,inputSizeHeight:224,inputSizeWidth:224,anchorOffsetX:.5,anchorOffsetY:.5,strides:[8,16,32,32,32],aspectRatios:[1],fixedAnchorSize:!0},fn={runtime:"tfjs",modelType:"full",enableSmoothing:!0,enableSegmentation:!1,smoothSegmentation:!0,detectorModelUrl:"https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/detector/1",landmarkModelUrl:"https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/landmark/full/2"},Gh={maxPoses:1,flipHorizontal:!1},qh={applyExponentialOnBoxSize:!1,flipVertically:!1,ignoreClasses:[],numClasses:1,numBoxes:2254,numCoords:12,boxCoordOffset:0,keypointCoordOffset:4,numKeypoints:4,numValuesPerKeypoint:2,sigmoidScore:!0,scoreClippingThresh:100,reverseOutputOrder:!0,xScale:224,yScale:224,hScale:224,wScale:224,minScoreThresh:.5},Kh=.3,Kr={shiftX:0,shiftY:0,scaleX:1.25,scaleY:1.25,squareLong:!0},Xh={outputTensorSize:{width:224,height:224},keepAspectRatio:!0,outputTensorFloatRange:[-1,1],borderMode:"zero"},Yh={outputTensorSize:{width:256,height:256},keepAspectRatio:!0,outputTensorFloatRange:[0,1],borderMode:"zero"},Qh={numLandmarks:39,inputImageWidth:256,inputImageHeight:256,visibilityActivation:"sigmoid",flipHorizontally:!1,flipVertically:!1},Jh={numLandmarks:39,inputImageWidth:1,inputImageHeight:1,visibilityActivation:"sigmoid",flipHorizontally:!1,flipVertically:!1},Zh={kernelSize:7,minConfidenceToRefine:.5},Xr={alpha:.1},ef={oneEuroFilter:{frequency:30,minCutOff:.05,beta:80,derivateCutOff:1,minAllowedObjectScale:1e-6}},tf={oneEuroFilter:{frequency:30,minCutOff:.01,beta:10,derivateCutOff:1,minAllowedObjectScale:1e-6}},nf={oneEuroFilter:{frequency:30,minCutOff:.1,beta:40,derivateCutOff:1,minAllowedObjectScale:1e-6,disableValueScaling:!0}},rf={activation:"none"},af={combineWithPreviousRatio:.7},sf=function(){function a(e){this.mask=e}return a.prototype.toCanvasImageSource=function(){return be(this,void 0,void 0,function(){return _e(this,function(e){return[2,Mi(this.mask)]})})},a.prototype.toImageData=function(){return be(this,void 0,void 0,function(){return _e(this,function(e){return[2,Ai(this.mask)]})})},a.prototype.toTensor=function(){return be(this,void 0,void 0,function(){return _e(this,function(e){return[2,this.mask]})})},a.prototype.getUnderlyingType=function(){return"tensor"},a}();function of(a){return Ti(a),"person"}var uf=function(){function a(e,r,i,s,t,u){this.detectorModel=e,this.landmarkModel=r,this.enableSmoothing=i,this.enableSegmentation=s,this.smoothSegmentation=t,this.modelType=u,this.regionOfInterest=null,this.prevFilteredSegmentationMask=null,this.anchors=function(f){f.reduceBoxesInLowestLayer==null&&(f.reduceBoxesInLowestLayer=!1),f.interpolatedScaleAspectRatio==null&&(f.interpolatedScaleAspectRatio=1),f.fixedAnchorSize==null&&(f.fixedAnchorSize=!1);for(var _=[],m=0;m<f.numLayers;){for(var y=[],b=[],k=[],S=[],I=m;I<f.strides.length&&f.strides[I]===f.strides[m];){var w=Hr(f.minScale,f.maxScale,I,f.strides.length);if(I===0&&f.reduceBoxesInLowestLayer)k.push(1),k.push(2),k.push(.5),S.push(.1),S.push(w),S.push(w);else{for(var A=0;A<f.aspectRatios.length;++A)k.push(f.aspectRatios[A]),S.push(w);if(f.interpolatedScaleAspectRatio>0){var O=I===f.strides.length-1?1:Hr(f.minScale,f.maxScale,I+1,f.strides.length);S.push(Math.sqrt(w*O)),k.push(f.interpolatedScaleAspectRatio)}}I++}for(var F=0;F<k.length;++F){var P=Math.sqrt(k[F]);y.push(S[F]/P),b.push(S[F]*P)}var V=0,D=0;if(f.featureMapHeight.length>0)V=f.featureMapHeight[m],D=f.featureMapWidth[m];else{var j=f.strides[m];V=Math.ceil(f.inputSizeHeight/j),D=Math.ceil(f.inputSizeWidth/j)}for(var G=0;G<V;++G)for(var X=0;X<D;++X)for(var $=0;$<y.length;++$){var le={xCenter:(X+f.anchorOffsetX)/D,yCenter:(G+f.anchorOffsetY)/V,width:0,height:0};f.fixedAnchorSize?(le.width=1,le.height=1):(le.width=b[$],le.height=y[$]),_.push(le)}m=I}return _}($h);var p=Mn(this.anchors.map(function(f){return f.width})),h=Mn(this.anchors.map(function(f){return f.height})),d=Mn(this.anchors.map(function(f){return f.xCenter})),n=Mn(this.anchors.map(function(f){return f.yCenter}));this.anchorTensor={x:d,y:n,w:p,h},this.prevFilteredSegmentationMask=this.enableSegmentation?rn([],[0,0]):null}return a.prototype.estimatePoses=function(e,r,i){return be(this,void 0,void 0,function(){var s,t,u,p,h,d,n,f,_,m,y,b,k,S,I,w,A,O,F,P,V,D,j;return _e(this,function(G){switch(G.label){case 0:return s=function(X){var $;if(($=X==null?Gh:xe({},X)).maxPoses==null&&($.maxPoses=1),$.maxPoses<=0)throw new Error("Invalid maxPoses "+$.maxPoses+". Should be > 0.");if($.maxPoses>1)throw new Error("Multi-pose detection is not implemented yet. Please set maxPoses to 1.");return $}(r),e==null?(this.reset(),[2,[]]):(this.maxPoses=s.maxPoses,this.timestamp=i!=null?1e3*i:Ci(e)?1e6*e.currentTime:null,t=Nn(e),u=Ye(function(){return Rn(Fr(e),"float32")}),(p=this.regionOfInterest)!=null?[3,2]:[4,this.detectPose(u)]);case 1:if((h=G.sent()).length===0)return this.reset(),u.dispose(),[2,[]];d=h[0],p=this.poseDetectionToRoi(d,t),G.label=2;case 2:return[4,this.poseLandmarksByRoi(p,u)];case 3:return n=G.sent(),u.dispose(),n==null?(this.reset(),[2,[]]):(f=n.landmarks,_=n.auxiliaryLandmarks,m=n.poseScore,y=n.worldLandmarks,b=n.segmentationMask,k=this.poseLandmarkFiltering(f,_,y,t),S=k.actualLandmarksFiltered,I=k.auxiliaryLandmarksFiltered,w=k.actualWorldLandmarksFiltered,A=this.poseLandmarksToRoi(I,t),this.regionOfInterest=A,O=this.smoothSegmentation&&b!=null?this.poseSegmentationFiltering(b):b,(F=S!=null?Oi(S,t):null)!=null&&F.forEach(function(X,$){X.name=En[$]}),(P=w)!=null&&P.forEach(function(X,$){X.name=En[$]}),V={score:m,keypoints:F,keypoints3D:P},O!==null&&(D=Ye(function(){var X=Tn(O,2),$=mr(X,[[0,0],[0,0],[0,1]]);return su($,[[0,0],[0,0],[0,2]],"symmetric")}),this.smoothSegmentation||Ze(O),j={maskValueToLabel:of,mask:new sf(D)},V.segmentation=j),[2,[V]])}})})},a.prototype.poseSegmentationFiltering=function(e){var r=this.prevFilteredSegmentationMask;return r.size===0?this.prevFilteredSegmentationMask=e:(this.prevFilteredSegmentationMask=Bh(r,e,af),Ze(e)),Ze(r),this.prevFilteredSegmentationMask},a.prototype.dispose=function(){this.detectorModel.dispose(),this.landmarkModel.dispose(),Ze([this.anchorTensor.x,this.anchorTensor.y,this.anchorTensor.w,this.anchorTensor.h,this.prevFilteredSegmentationMask])},a.prototype.reset=function(){this.regionOfInterest=null,this.enableSegmentation&&(Ze(this.prevFilteredSegmentationMask),this.prevFilteredSegmentationMask=rn([],[0,0])),this.visibilitySmoothingFilterActual=null,this.visibilitySmoothingFilterAuxiliary=null,this.landmarksSmoothingFilterActual=null,this.landmarksSmoothingFilterAuxiliary=null},a.prototype.detectPose=function(e){return be(this,void 0,void 0,function(){var r,i,s,t,u,p,h,d,n,f;return _e(this,function(_){switch(_.label){case 0:return r=kr(e,Xh),i=r.imageTensor,s=r.padding,t=this.detectorModel.predict(i),u=Ph(t),p=u.boxes,[4,jh([h=u.logits,p],this.anchorTensor,qh)];case 1:return(d=_.sent()).length===0?(Ze([i,t,h,p]),[2,d]):[4,Dh(d,this.maxPoses,Kh)];case 2:return n=_.sent(),f=function(m,y){m===void 0&&(m=[]);for(var b=y.left,k=y.top,S=y.left+y.right,I=y.top+y.bottom,w=0;w<m.length;w++){var A=m[w],O=A.locationData.relativeBoundingBox,F=(O.xMin-b)/(1-S),P=(O.yMin-k)/(1-I),V=O.width/(1-S),D=O.height/(1-I);O.xMin=F,O.yMin=P,O.width=V,O.height=D,O.xMax=F+V,O.yMax=P+D;var j=A.locationData.relativeKeypoints;j&&j.forEach(function(G){var X=(G.x-b)/(1-S),$=(G.y-k)/(1-I);G.x=X,G.y=$})}return m}(n,s),Ze([i,t,h,p]),[2,f]}})})},a.prototype.poseDetectionToRoi=function(e,r){return Gr(ir(e,r,{rotationVectorEndKeypointIndex:1,rotationVectorStartKeypointIndex:0,rotationVectorTargetAngleDegree:90}),r,Kr)},a.prototype.poseLandmarksByRoi=function(e,r){return be(this,void 0,void 0,function(){var i,s,t,u,p,h,d,n,f,_,m,y,b,k;return _e(this,function(S){switch(S.label){case 0:if(i=Nn(r),s=kr(r,Yh,e),t=s.imageTensor,u=s.padding,p=s.transformationMatrix,this.modelType!=="lite"&&this.modelType!=="full"&&this.modelType!=="heavy")throw new Error("Model type must be one of lite, full or heavy,but got "+this.modelType);return h=["ld_3d","output_poseflag","activation_heatmap","world_3d"],this.enableSegmentation&&h.push("activation_segmentation"),d=this.landmarkModel.execute(t,h),[4,this.tensorsToPoseLandmarksAndSegmentation(d)];case 1:return(n=S.sent())==null?(Ze(d),Ze(t),[2,null]):(f=n.landmarks,_=n.auxiliaryLandmarks,m=n.poseScore,y=n.worldLandmarks,b=n.segmentationMask,[4,this.poseLandmarksAndSegmentationInverseProjection(i,e,u,p,f,_,y,b)]);case 2:return k=S.sent(),Ze(d),Ze(t),[2,xe({poseScore:m},k)]}})})},a.prototype.poseLandmarksAndSegmentationInverseProjection=function(e,r,i,s,t,u,p,h){return be(this,void 0,void 0,function(){var d,n,f,_,m,y;return _e(this,function(b){return d=zr(t,i),n=zr(u,i),f=Wr(d,r),_=Wr(n,r),m=function(k,S){for(var I=[],w=0,A=k;w<A.length;w++){var O=A[w],F=O.x,P=O.y,V=S.rotation,D=Math.cos(V)*F-Math.sin(V)*P,j=Math.sin(V)*F+Math.cos(V)*P,G=xe({},O);G.x=D,G.y=j,I.push(G)}return I}(p,r),y=null,this.enableSegmentation&&(y=Ye(function(){var k=h.shape,S=k[0],I=k[1],w=function(F){var P=Ei(new Array(16).fill(0));P[0][0]=Xe(F,0,0),P[1][0]=-Xe(F,0,1),P[2][0]=Xe(F,0,2),P[3][0]=-Xe(F,0,3),P[0][2]=Xe(F,2,0),P[1][2]=-Xe(F,2,1),P[2][2]=Xe(F,2,2),P[3][2]=-Xe(F,2,3),P[0][1]=-Xe(F,1,0),P[1][1]=Xe(F,1,1),P[2][1]=-Xe(F,1,2),P[3][1]=Xe(F,1,3),P[0][3]=-Xe(F,3,0),P[1][3]=Xe(F,3,1),P[2][3]=-Xe(F,3,2),P[3][3]=Xe(F,3,3);for(var V=F[0][0]*P[0][0]+F[1][0]*P[0][1]+F[2][0]*P[0][2]+F[3][0]*P[0][3],D=0;D<P.length;D++)for(var j=0;j<P.length;j++)P[D][j]/=V;return P}(s),A=rn(Fi(w,{width:I,height:S},e),[1,8]),O=[1,S,I,1];return Re(an.transform(Vt(h,O),A,"bilinear","constant",0,[e.height,e.width]),[0,3])}),Ze(h)),[2,{landmarks:f,auxiliaryLandmarks:_,worldLandmarks:m,segmentationMask:y}]})})},a.prototype.tensorsToPoseLandmarksAndSegmentation=function(e){return be(this,void 0,void 0,function(){var r,i,s,t,u,p,h,d,n,f,_,m,y;return _e(this,function(b){switch(b.label){case 0:return r=e[0],i=e[1],s=e[2],t=e[3],u=this.enableSegmentation?e[4]:null,[4,i.data()];case 1:return(p=b.sent()[0])<.5?[2,null]:[4,$r(r,Qh)];case 2:return[4,Lh(b.sent(),s,Zh)];case 3:return h=b.sent(),d=h.slice(0,33),n=h.slice(33,35),[4,$r(t,Jh)];case 4:return f=b.sent(),_=f.slice(0,33),m=function(k,S,I){I===void 0&&(I=!0);for(var w=[],A=0;A<k.length;A++){var O=xe({},S[A]);I&&(O.score=k[A].score),w.push(O)}return w}(d,_,!0),y=this.enableSegmentation?function(k,S,I){return Ye(function(){var w=Re(k,[0]),A=w.shape[2];if(A===1){var O=w;switch(S.activation){case"none":break;case"sigmoid":O=Tr(O);break;case"softmax":throw new Error("Softmax activation requires two channels.");default:throw new Error("Activation not supported ("+S.activation+")")}var F=I?an.resizeBilinear(O,[I.height,I.width]):O;return Re(F,[2])}throw new Error("Unsupported number of tensor channels "+A)})}(u,rf):null,[2,{landmarks:d,auxiliaryLandmarks:n,poseScore:p,worldLandmarks:m,segmentationMask:y}]}})})},a.prototype.poseLandmarksToRoi=function(e,r){return Gr(ir(Ur(e),r,{rotationVectorStartKeypointIndex:0,rotationVectorEndKeypointIndex:1,rotationVectorTargetAngleDegree:90}),r,Kr)},a.prototype.poseLandmarkFiltering=function(e,r,i,s){var t,u,p;if(this.timestamp!=null&&this.enableSmoothing){var h=ir(Ur(r),s,{rotationVectorEndKeypointIndex:0,rotationVectorStartKeypointIndex:1,rotationVectorTargetAngleDegree:90});this.visibilitySmoothingFilterActual==null&&(this.visibilitySmoothingFilterActual=new qr(Xr)),t=this.visibilitySmoothingFilterActual.apply(e),this.visibilitySmoothingFilterAuxiliary==null&&(this.visibilitySmoothingFilterAuxiliary=new qr(Xr)),u=this.visibilitySmoothingFilterAuxiliary.apply(r),p=this.visibilitySmoothingFilterActual.apply(i),this.landmarksSmoothingFilterActual==null&&(this.landmarksSmoothingFilterActual=new cr(ef)),t=this.landmarksSmoothingFilterActual.apply(t,this.timestamp,s,!0,h),this.landmarksSmoothingFilterAuxiliary==null&&(this.landmarksSmoothingFilterAuxiliary=new cr(tf)),u=this.landmarksSmoothingFilterAuxiliary.apply(u,this.timestamp,s,!0,h),this.worldLandmarksSmoothingFilterActual==null&&(this.worldLandmarksSmoothingFilterActual=new cr(nf)),p=this.worldLandmarksSmoothingFilterActual.apply(i,this.timestamp)}else t=e,u=r,p=i;return{actualLandmarksFiltered:t,auxiliaryLandmarksFiltered:u,actualWorldLandmarksFiltered:p}},a}();function lf(a){return be(this,void 0,void 0,function(){var e,r,i,s,t,u;return _e(this,function(p){switch(p.label){case 0:return e=function(h){var d=xe({},h??fn);if(d.enableSmoothing==null&&(d.enableSmoothing=fn.enableSmoothing),d.enableSegmentation==null&&(d.enableSegmentation=fn.enableSegmentation),d.smoothSegmentation==null&&(d.smoothSegmentation=fn.smoothSegmentation),d.modelType==null&&(d.modelType=fn.modelType),d.detectorModelUrl==null&&(d.detectorModelUrl=fn.detectorModelUrl),d.landmarkModelUrl==null)switch(d.modelType){case"lite":d.landmarkModelUrl="https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/landmark/lite/2";break;case"heavy":d.landmarkModelUrl="https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/landmark/heavy/2";break;case"full":default:d.landmarkModelUrl="https://tfhub.dev/mediapipe/tfjs-model/blazepose_3d/landmark/full/2"}return d}(a),r=typeof e.detectorModelUrl=="string"&&e.detectorModelUrl.indexOf("https://tfhub.dev")>-1,i=typeof e.landmarkModelUrl=="string"&&e.landmarkModelUrl.indexOf("https://tfhub.dev")>-1,[4,Promise.all([gn(e.detectorModelUrl,{fromTFHub:r}),gn(e.landmarkModelUrl,{fromTFHub:i})])];case 1:return s=p.sent(),t=s[0],u=s[1],[2,new uf(t,u,e.enableSmoothing,e.enableSegmentation,e.smoothSegmentation,e.modelType)]}})})}var sn,It,Pi=function(){function a(e){(function(r){if(r.maxTracks<1)throw new Error("Must specify 'maxTracks' to be at least 1, but encountered "+r.maxTracks);if(r.maxAge<=0)throw new Error("Must specify 'maxAge' to be positive, but encountered "+r.maxAge);if(r.keypointTrackerParams!==void 0){if(r.keypointTrackerParams.keypointConfidenceThreshold<0||r.keypointTrackerParams.keypointConfidenceThreshold>1)throw new Error("Must specify 'keypointConfidenceThreshold' to be in the range [0, 1], but encountered "+r.keypointTrackerParams.keypointConfidenceThreshold);if(r.keypointTrackerParams.minNumberOfKeypoints<1)throw new Error("Must specify 'minNumberOfKeypoints' to be at least 1, but encountered "+r.keypointTrackerParams.minNumberOfKeypoints);for(var i=0,s=r.keypointTrackerParams.keypointFalloff;i<s.length;i++){var t=s[i];if(t<=0)throw new Error("Must specify each keypoint falloff parameterto be positive but encountered "+t)}}})(e),this.tracks=[],this.maxTracks=e.maxTracks,this.maxAge=1e3*e.maxAge,this.minSimilarity=e.minSimilarity,this.nextID=1}return a.prototype.apply=function(e,r){this.filterOldTracks(r);var i=this.computeSimilarity(e);return this.assignTracks(e,i,r),this.updateTracks(r),e},a.prototype.getTracks=function(){return this.tracks.slice()},a.prototype.getTrackIDs=function(){return new Set(this.tracks.map(function(e){return e.id}))},a.prototype.filterOldTracks=function(e){var r=this;this.tracks=this.tracks.filter(function(i){return e-i.lastTimestamp<=r.maxAge})},a.prototype.assignTracks=function(e,r,i){for(var s=Array.from(Array(r[0].length).keys()),t=[],u=0,p=Array.from(Array(e.length).keys());u<p.length;u++){var h=p[u];if(s.length!==0){for(var d=-1,n=-1,f=0,_=s;f<_.length;f++){var m=_[f],y=r[h][m];y>=this.minSimilarity&&y>n&&(d=m,n=y)}if(d>=0){var b=this.tracks[d];b=Object.assign(b,this.createTrack(e[h],i,b.id)),e[h].id=b.id;var k=s.indexOf(d);s.splice(k,1)}else t.push(h)}else t.push(h)}for(var S=0,I=t;S<I.length;S++){h=I[S];var w=this.createTrack(e[h],i);this.tracks.push(w),e[h].id=w.id}},a.prototype.updateTracks=function(e){this.tracks.sort(function(r,i){return i.lastTimestamp-r.lastTimestamp}),this.tracks=this.tracks.slice(0,this.maxTracks)},a.prototype.createTrack=function(e,r,i){var s={id:i||this.nextTrackID(),lastTimestamp:r,keypoints:nn(e.keypoints).map(function(t){return xe({},t)})};return e.box!==void 0&&(s.box=xe({},e.box)),s},a.prototype.nextTrackID=function(){var e=this.nextID;return this.nextID+=1,e},a.prototype.remove=function(){for(var e=[],r=0;r<arguments.length;r++)e[r]=arguments[r];this.tracks=this.tracks.filter(function(i){return!e.includes(i.id)})},a.prototype.reset=function(){this.tracks=[]},a}(),cf=function(a){function e(r){return a.call(this,r)||this}return xi(e,a),e.prototype.computeSimilarity=function(r){var i=this;return r.length===0||this.tracks.length===0?[[]]:r.map(function(s){return i.tracks.map(function(t){return i.iou(s,t)})})},e.prototype.iou=function(r,i){var s=Math.max(r.box.xMin,i.box.xMin),t=Math.max(r.box.yMin,i.box.yMin),u=Math.min(r.box.xMax,i.box.xMax),p=Math.min(r.box.yMax,i.box.yMax);if(s>=u||t>=p)return 0;var h=(u-s)*(p-t);return h/(r.box.width*r.box.height+i.box.width*i.box.height-h)},e}(Pi),pf=function(a){function e(r){var i=a.call(this,r)||this;return i.keypointThreshold=r.keypointTrackerParams.keypointConfidenceThreshold,i.keypointFalloff=r.keypointTrackerParams.keypointFalloff,i.minNumKeyoints=r.keypointTrackerParams.minNumberOfKeypoints,i}return xi(e,a),e.prototype.computeSimilarity=function(r){if(r.length===0||this.tracks.length===0)return[[]];for(var i=[],s=0,t=r;s<t.length;s++){for(var u=t[s],p=[],h=0,d=this.tracks;h<d.length;h++){var n=d[h];p.push(this.oks(u,n))}i.push(p)}return i},e.prototype.oks=function(r,i){for(var s=this.area(i.keypoints)+1e-6,t=0,u=0,p=0;p<r.keypoints.length;++p){var h=r.keypoints[p],d=i.keypoints[p];if(!(h.score<this.keypointThreshold||d.score<this.keypointThreshold)){u+=1;var n=Math.pow(h.x-d.x,2)+Math.pow(h.y-d.y,2),f=2*this.keypointFalloff[p];t+=Math.exp(-1*n/(2*s*Math.pow(f,2)))}}return u<this.minNumKeyoints?0:t/u},e.prototype.area=function(r){var i=this,s=r.filter(function(h){return h.score>i.keypointThreshold}),t=Math.min.apply(Math,nn([1],s.map(function(h){return h.x}))),u=Math.max.apply(Math,nn([0],s.map(function(h){return h.x}))),p=Math.min.apply(Math,nn([1],s.map(function(h){return h.y})));return(u-t)*(Math.max.apply(Math,nn([0],s.map(function(h){return h.y})))-p)},e}(Pi);function df(a){switch(a){case It.BlazePose:return En.reduce(function(e,r,i){return e[r]=i,e},{});case It.PoseNet:case It.MoveNet:return St.reduce(function(e,r,i){return e[r]=i,e},{});default:throw new Error("Model "+a+" is not supported.")}}(function(a){a.Keypoint="keypoint",a.BoundingBox="boundingBox"})(sn||(sn={})),function(a){a.MoveNet="MoveNet",a.BlazePose="BlazePose",a.PoseNet="PoseNet"}(It||(It={}));var Yr=["SinglePose.Lightning","SinglePose.Thunder","MultiPose.Lightning"],Di={modelType:"SinglePose.Lightning",enableSmoothing:!0},Qr={},Jr={frequency:30,minCutOff:2.5,beta:300,derivateCutOff:2.5,thresholdCutOff:.5,thresholdBeta:5,disableValueScaling:!0},pr={maxTracks:18,maxAge:1e3,minSimilarity:.2,keypointTrackerParams:{keypointConfidenceThreshold:.3,keypointFalloff:[.026,.025,.025,.035,.035,.079,.079,.072,.072,.062,.062,.107,.107,.087,.087,.089,.089],minNumberOfKeypoints:4}},Zr={maxTracks:18,maxAge:1e3,minSimilarity:.15,trackerParams:{}};function hf(a,e,r,i){for(var s={},t=0,u=St;t<u.length;t++){var p=u[t];s[p]=[e[r[p]].y*i.height,e[r[p]].x*i.width]}if(function(I,w){return(I[w.left_hip].score>.2||I[w.right_hip].score>.2)&&(I[w.left_shoulder].score>.2||I[w.right_shoulder].score>.2)}(e,r)){var h=(s.left_hip[0]+s.right_hip[0])/2,d=(s.left_hip[1]+s.right_hip[1])/2,n=function(I,w,A,O,F){for(var P=["left_shoulder","right_shoulder","left_hip","right_hip"],V=0,D=0,j=0;j<P.length;j++)(Q=Math.abs(O-A[P[j]][0]))>V&&(V=Q),(ce=Math.abs(F-A[P[j]][1]))>D&&(D=ce);for(var G=0,X=0,$=0,le=Object.keys(A);$<le.length;$++){var Q,ce,ke=le[$];I[w[ke]].score<.2||((Q=Math.abs(O-A[ke][0]))>G&&(G=Q),(ce=Math.abs(F-A[ke][1]))>X&&(X=ce))}return[V,D,G,X]}(e,r,s,h,d),f=n[0],_=n[1],m=n[2],y=n[3],b=Math.max(1.9*_,1.9*f,1.2*m,1.2*y),k=[h-(b=Math.min(b,Math.max(d,i.width-d,h,i.height-h))),d-b];if(b>Math.max(i.width,i.height)/2)return Sr(a==null,i);var S=2*b;return{yMin:k[0]/i.height,xMin:k[1]/i.width,yMax:(k[0]+S)/i.height,xMax:(k[1]+S)/i.width,height:(k[0]+S)/i.height-k[0]/i.height,width:(k[1]+S)/i.width-k[1]/i.width}}return Sr(a==null,i)}function Sr(a,e){var r,i,s,t;return a?e.width>e.height?(r=1,i=e.height/e.width,s=0,t=(e.width/2-e.height/2)/e.width):(r=e.width/e.height,i=1,s=(e.height/2-e.width/2)/e.height,t=0):e.width>e.height?(r=e.width/e.height,i=1,s=(e.height/2-e.width/2)/e.height,t=0):(r=1,i=e.height/e.width,s=0,t=(e.width/2-e.height/2)/e.width),{yMin:s,xMin:t,yMax:s+r,xMax:t+i,height:r,width:i}}function ff(a){var e,r=a==null?Di:xe({},a);if(r.modelType==null)r.modelType="SinglePose.Lightning";else if(Yr.indexOf(r.modelType)<0)throw new Error("Invalid architecture "+r.modelType+". Should be one of "+Yr);if(r.enableSmoothing==null&&(r.enableSmoothing=!0),r.minPoseScore!=null&&(r.minPoseScore<0||r.minPoseScore>1))throw new Error("minPoseScore should be between 0.0 and 1.0");if(r.multiPoseMaxDimension!=null&&(r.multiPoseMaxDimension%32!=0||r.multiPoseMaxDimension<32))throw new Error("multiPoseMaxDimension must be a multiple of 32 and higher than 0");if(r.modelType==="MultiPose.Lightning"&&r.enableTracking==null&&(r.enableTracking=!0),r.modelType==="MultiPose.Lightning"&&r.enableTracking===!0)if(r.trackerType==null&&(r.trackerType=sn.BoundingBox),r.trackerType===sn.Keypoint)r.trackerConfig!=null?r.trackerConfig=function(i){var s=ea(pr,i);return s.keypointTrackerParams=xe({},pr.keypointTrackerParams),i.keypointTrackerParams!=null&&(i.keypointTrackerParams.keypointConfidenceThreshold!=null&&(s.keypointTrackerParams.keypointConfidenceThreshold=i.keypointTrackerParams.keypointConfidenceThreshold),i.keypointTrackerParams.keypointFalloff!=null&&(s.keypointTrackerParams.keypointFalloff=i.keypointTrackerParams.keypointFalloff),i.keypointTrackerParams.minNumberOfKeypoints!=null&&(s.keypointTrackerParams.minNumberOfKeypoints=i.keypointTrackerParams.minNumberOfKeypoints)),s}(r.trackerConfig):r.trackerConfig=pr;else{if(r.trackerType!==sn.BoundingBox)throw new Error("Tracker type not supported by MoveNet");r.trackerConfig!=null?r.trackerConfig=(e=r.trackerConfig,ea(Zr,e)):r.trackerConfig=Zr}return r}function ea(a,e){var r={maxTracks:a.maxTracks,maxAge:a.maxAge,minSimilarity:a.minSimilarity};return e.maxTracks!=null&&(r.maxTracks=e.maxTracks),e.maxAge!=null&&(r.maxAge=e.maxAge),e.minSimilarity!=null&&(r.minSimilarity=e.minSimilarity),r}var mf=function(){function a(e,r){this.moveNetModel=e,this.modelInputResolution={height:0,width:0},this.keypointIndexByName=df(It.MoveNet),r.modelType==="SinglePose.Lightning"?(this.modelInputResolution.width=192,this.modelInputResolution.height=192):r.modelType==="SinglePose.Thunder"&&(this.modelInputResolution.width=256,this.modelInputResolution.height=256),this.multiPoseModel=r.modelType==="MultiPose.Lightning",this.multiPoseModel||(this.keypointFilter=new Ir(Jr),this.cropRegionFilterYMin=new Wt(.9),this.cropRegionFilterXMin=new Wt(.9),this.cropRegionFilterYMax=new Wt(.9),this.cropRegionFilterXMax=new Wt(.9)),this.enableSmoothing=r.enableSmoothing,r.minPoseScore?this.minPoseScore=r.minPoseScore:this.minPoseScore=.25,r.multiPoseMaxDimension?this.multiPoseMaxDimension=r.multiPoseMaxDimension:this.multiPoseMaxDimension=256,this.enableTracking=r.enableTracking,this.multiPoseModel&&this.enableTracking&&(r.trackerType===sn.Keypoint?this.tracker=new pf(r.trackerConfig):r.trackerType===sn.BoundingBox&&(this.tracker=new cf(r.trackerConfig)),this.enableSmoothing&&(this.keypointFilterMap=new Map))}return a.prototype.runSinglePersonPoseModel=function(e){return be(this,void 0,void 0,function(){var r,i,s,t,u;return _e(this,function(p){switch(p.label){case 0:if((r=this.moveNetModel.execute(e)).shape.length!==4||r.shape[0]!==1||r.shape[1]!==1||r.shape[2]!==17||r.shape[3]!==3)throw r.dispose(),new Error("Unexpected output shape from model: ["+r.shape+"]");return Un()==="webgpu"?[3,1]:(i=r.dataSync(),[3,3]);case 1:return[4,r.data()];case 2:i=p.sent(),p.label=3;case 3:for(r.dispose(),s={keypoints:[],score:0},t=0,u=0;u<17;++u)s.keypoints[u]={y:i[3*u],x:i[3*u+1],score:i[3*u+2]},s.keypoints[u].score>.2&&(++t,s.score+=s.keypoints[u].score);return t>0&&(s.score/=t),[2,s]}})})},a.prototype.runMultiPersonPoseModel=function(e){return be(this,void 0,void 0,function(){var r,i,s,t,u,p,h,d;return _e(this,function(n){switch(n.label){case 0:if((r=this.moveNetModel.execute(e)).shape.length!==3||r.shape[0]!==1||r.shape[2]!==56)throw r.dispose(),new Error("Unexpected output shape from model: ["+r.shape+"]");return Un()==="webgpu"?[3,1]:(i=r.dataSync(),[3,3]);case 1:return[4,r.data()];case 2:i=n.sent(),n.label=3;case 3:for(r.dispose(),s=[],t=i.length/56,u=0;u<t;++u)for(s[u]={keypoints:[]},p=56*u+51,s[u].box={yMin:i[p],xMin:i[p+1],yMax:i[p+2],xMax:i[p+3],width:i[p+3]-i[p+1],height:i[p+2]-i[p]},h=56*u+55,s[u].score=i[h],s[u].keypoints=[],d=0;d<17;++d)s[u].keypoints[d]={y:i[56*u+3*d],x:i[56*u+3*d+1],score:i[56*u+3*d+2]};return[2,s]}})})},a.prototype.estimatePoses=function(e,r,i){return r===void 0&&(r=Qr),be(this,void 0,void 0,function(){var s,t,u,p,h,d;return _e(this,function(n){switch(n.label){case 0:return r=function(f){return f==null?Qr:xe({},f)}(r),e==null?(this.reset(),[2,[]]):(i==null?Ci(e)&&(i=1e6*e.currentTime):i*=1e3,s=Fr(e),t=Nn(s),u=Tn(s,0),e instanceof bn||s.dispose(),p=[],this.multiPoseModel?[3,2]:[4,this.estimateSinglePose(u,t,i)]);case 1:return p=n.sent(),[3,4];case 2:return[4,this.estimateMultiplePoses(u,t,i)];case 3:p=n.sent(),n.label=4;case 4:for(h=0;h<p.length;++h)for(d=0;d<p[h].keypoints.length;++d)p[h].keypoints[d].name=St[d],p[h].keypoints[d].y*=t.height,p[h].keypoints[d].x*=t.width;return[2,p]}})})},a.prototype.estimateSinglePose=function(e,r,i){return be(this,void 0,void 0,function(){var s,t,u,p,h=this;return _e(this,function(d){switch(d.label){case 0:return this.cropRegion||(this.cropRegion=Sr(this.cropRegion==null,r)),s=Ye(function(){var n=rn([[h.cropRegion.yMin,h.cropRegion.xMin,h.cropRegion.yMax,h.cropRegion.xMax]]),f=iu([1],"int32"),_=[h.modelInputResolution.height,h.modelInputResolution.width];return Rn(an.cropAndResize(e,n,f,_,"bilinear",0),"int32")}),e.dispose(),[4,this.runSinglePersonPoseModel(s)];case 1:if(t=d.sent(),s.dispose(),t.score<this.minPoseScore)return this.reset(),[2,[]];for(u=0;u<t.keypoints.length;++u)t.keypoints[u].y=this.cropRegion.yMin+t.keypoints[u].y*this.cropRegion.height,t.keypoints[u].x=this.cropRegion.xMin+t.keypoints[u].x*this.cropRegion.width;return i!=null&&this.enableSmoothing&&(t.keypoints=this.keypointFilter.apply(t.keypoints,i,1)),p=hf(this.cropRegion,t.keypoints,this.keypointIndexByName,r),this.cropRegion=this.filterCropRegion(p),[2,[t]]}})})},a.prototype.estimateMultiplePoses=function(e,r,i){return be(this,void 0,void 0,function(){var s,t,u,p,h,d,n,f,_,m,y,b=this;return _e(this,function(k){switch(k.label){case 0:return r.width>r.height?(t=this.multiPoseMaxDimension,u=Math.round(this.multiPoseMaxDimension*r.height/r.width),s=an.resizeBilinear(e,[u,t]),h=t,d=32*Math.ceil(u/32),p=mr(s,[[0,0],[0,d-u],[0,0],[0,0]])):(t=Math.round(this.multiPoseMaxDimension*r.width/r.height),u=this.multiPoseMaxDimension,s=an.resizeBilinear(e,[u,t]),h=32*Math.ceil(t/32),d=u,p=mr(s,[[0,0],[0,0],[0,h-t],[0,0]])),s.dispose(),e.dispose(),n=Rn(p,"int32"),p.dispose(),[4,this.runMultiPersonPoseModel(n)];case 1:for(f=k.sent(),n.dispose(),f=f.filter(function(S){return S.score>=b.minPoseScore}),m=0;m<f.length;++m)for(_=0;_<f[m].keypoints.length;++_)f[m].keypoints[_].y*=d/u,f[m].keypoints[_].x*=h/t;if(this.enableTracking&&(this.tracker.apply(f,i),this.enableSmoothing)){for(m=0;m<f.length;++m)this.keypointFilterMap.has(f[m].id)||this.keypointFilterMap.set(f[m].id,new Ir(Jr)),f[m].keypoints=this.keypointFilterMap.get(f[m].id).apply(f[m].keypoints,i,1);y=this.tracker.getTrackIDs(),this.keypointFilterMap.forEach(function(S,I){y.has(I)||b.keypointFilterMap.delete(I)})}return[2,f]}})})},a.prototype.filterCropRegion=function(e){if(e){var r=this.cropRegionFilterYMin.apply(e.yMin),i=this.cropRegionFilterXMin.apply(e.xMin),s=this.cropRegionFilterYMax.apply(e.yMax),t=this.cropRegionFilterXMax.apply(e.xMax);return{yMin:r,xMin:i,yMax:s,xMax:t,height:s-r,width:t-i}}return this.cropRegionFilterYMin.reset(),this.cropRegionFilterXMin.reset(),this.cropRegionFilterYMax.reset(),this.cropRegionFilterXMax.reset(),null},a.prototype.dispose=function(){this.moveNetModel.dispose()},a.prototype.reset=function(){this.cropRegion=null,this.resetFilters()},a.prototype.resetFilters=function(){this.keypointFilter.reset(),this.cropRegionFilterYMin.reset(),this.cropRegionFilterXMin.reset(),this.cropRegionFilterYMax.reset(),this.cropRegionFilterXMax.reset()},a}();function gf(a){return a===void 0&&(a=Di),be(this,void 0,void 0,function(){var e,r,i,s;return _e(this,function(t){switch(t.label){case 0:return e=ff(a),i=!0,e.modelUrl?(i=typeof e.modelUrl=="string"&&e.modelUrl.indexOf("https://tfhub.dev")>-1,[4,gn(e.modelUrl,{fromTFHub:i})]):[3,2];case 1:return r=t.sent(),[3,4];case 2:return s=void 0,e.modelType==="SinglePose.Lightning"?s="https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4":e.modelType==="SinglePose.Thunder"?s="https://tfhub.dev/google/tfjs-model/movenet/singlepose/thunder/4":e.modelType==="MultiPose.Lightning"&&(s="https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1"),[4,gn(s,{fromTFHub:i})];case 3:r=t.sent(),t.label=4;case 4:return Un()==="webgl"&&Hn().set("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD",0),[2,new mf(r,e)]}})})}var ta={architecture:"MobileNetV1",outputStride:16,multiplier:.75,inputResolution:{height:257,width:257}},na=["MobileNetV1","ResNet50"],ra={MobileNetV1:[8,16],ResNet50:[16]},yf=[8,16,32],aa={MobileNetV1:[.5,.75,1],ResNet50:[1]},sa=[1,2,4],bf={maxPoses:1,flipHorizontal:!1},_f={maxPoses:5,flipHorizontal:!1,scoreThreshold:.5,nmsRadius:20},vf=[-123.15,-115.9,-103.06];function dr(a){return Math.floor(a/2)}var wf=function(){function a(e,r){this.priorityQueue=new Array(e),this.numberOfElements=-1,this.getElementValue=r}return a.prototype.enqueue=function(e){this.priorityQueue[++this.numberOfElements]=e,this.swim(this.numberOfElements)},a.prototype.dequeue=function(){var e=this.priorityQueue[0];return this.exchange(0,this.numberOfElements--),this.sink(0),this.priorityQueue[this.numberOfElements+1]=null,e},a.prototype.empty=function(){return this.numberOfElements===-1},a.prototype.size=function(){return this.numberOfElements+1},a.prototype.all=function(){return this.priorityQueue.slice(0,this.numberOfElements+1)},a.prototype.max=function(){return this.priorityQueue[0]},a.prototype.swim=function(e){for(;e>0&&this.less(dr(e),e);)this.exchange(e,dr(e)),e=dr(e)},a.prototype.sink=function(e){for(;2*e<=this.numberOfElements;){var r=2*e;if(r<this.numberOfElements&&this.less(r,r+1)&&r++,!this.less(e,r))break;this.exchange(e,r),e=r}},a.prototype.getValueAt=function(e){return this.getElementValue(this.priorityQueue[e])},a.prototype.less=function(e,r){return this.getValueAt(e)<this.getValueAt(r)},a.prototype.exchange=function(e,r){var i=this.priorityQueue[e];this.priorityQueue[e]=this.priorityQueue[r],this.priorityQueue[r]=i},a}();function kf(a,e,r,i,s,t){for(var u=t.shape,p=u[0],h=u[1],d=!0,n=Math.max(r-s,0),f=Math.min(r+s+1,p),_=n;_<f;++_){for(var m=Math.max(i-s,0),y=Math.min(i+s+1,h),b=m;b<y;++b)if(t.get(_,b,a)>e){d=!1;break}if(!d)break}return d}function If(a){return be(this,void 0,void 0,function(){return _e(this,function(e){return[2,Promise.all(a.map(function(r){return r.buffer()}))]})})}function Li(a,e,r,i){return{y:i.get(a,e,r),x:i.get(a,e,r+17)}}function Bi(a,e,r){var i=Li(a.heatmapY,a.heatmapX,a.id,r),s=i.y,t=i.x;return{x:a.heatmapX*e+t,y:a.heatmapY*e+s}}function ji(a,e,r,i){var s=r.x,t=r.y;return a.some(function(u){var p,h,d,n,f,_,m=u.keypoints;return p=t,h=s,d=m[i].y,n=m[i].x,(f=d-p)*f+(_=n-h)*_<=e})}var ia=St.reduce(function(a,e,r){return a[e]=r,a},{}),Vi=[["nose","left_eye"],["left_eye","left_ear"],["nose","right_eye"],["right_eye","right_ear"],["nose","left_shoulder"],["left_shoulder","left_elbow"],["left_elbow","left_wrist"],["left_shoulder","left_hip"],["left_hip","left_knee"],["left_knee","left_ankle"],["nose","right_shoulder"],["right_shoulder","right_elbow"],["right_elbow","right_wrist"],["right_shoulder","right_hip"],["right_hip","right_knee"],["right_knee","right_ankle"]].map(function(a){var e=a[0],r=a[1];return[ia[e],ia[r]]}),hr=Vi.map(function(a){return a[1]}),oa=Vi.map(function(a){return a[0]});function ua(a,e,r){return a<e?e:a>r?r:a}function fr(a,e,r,i){return{y:ua(Math.round(a.y/e),0,r-1),x:ua(Math.round(a.x/e),0,i-1)}}function la(a,e){return{x:a.x+e.x,y:a.y+e.y}}function ca(a,e,r,i,s,t,u,p){p===void 0&&(p=2);for(var h=i.shape,d=h[0],n=h[1],f={y:e.y,x:e.x},_=la(f,function(I,w,A){var O=A.shape[2]/2;return{y:A.get(w.y,w.x,I),x:A.get(w.y,w.x,O+I)}}(a,fr(f,t,d,n),u)),m=0;m<p;m++){var y=fr(_,t,d,n),b=Li(y.y,y.x,r,s);_=la({x:y.x*t,y:y.y*t},{x:b.x,y:b.y})}var k=fr(_,t,d,n),S=i.get(k.y,k.x,r);return{y:_.y,x:_.x,name:St[r],score:S}}function Sf(a,e,r,i,s,t){var u=e.shape[2],p=hr.length,h=new Array(u),d=a.part,n=a.score,f=Bi(d,i,r);h[d.id]={score:n,name:St[d.id],y:f.y,x:f.x};for(var _=p-1;_>=0;--_){var m=hr[_],y=oa[_];h[m]&&!h[y]&&(h[y]=ca(_,h[m],y,e,r,i,t))}for(_=0;_<p;++_)m=oa[_],y=hr[_],h[m]&&!h[y]&&(h[y]=ca(_,h[m],y,e,r,i,s));return h}function xf(a,e,r){return r.reduce(function(i,s,t){var u=s.y,p=s.x,h=s.score;return ji(a,e,{y:u,x:p},t)||(i+=h),i},0)/r.length}function Mf(a,e,r,i,s,t,u,p){return u===void 0&&(u=.5),p===void 0&&(p=20),be(this,void 0,void 0,function(){var h,d,n,f,_,m,y,b,k,S,I,w;return _e(this,function(A){switch(A.label){case 0:return[4,If([a,e,r,i])];case 1:for(h=A.sent(),d=h[0],n=h[1],f=h[2],_=h[3],m=[],y=function(O,F,P){for(var V=P.shape,D=V[0],j=V[1],G=V[2],X=new wf(D*j*G,function(ke){return ke.score}),$=0;$<D;++$)for(var le=0;le<j;++le)for(var Q=0;Q<G;++Q){var ce=P.get($,le,Q);ce<O||kf(Q,ce,$,le,F,P)&&X.enqueue({score:ce,part:{heatmapY:$,heatmapX:le,id:Q}})}return X}(u,1,d),b=p*p;m.length<t&&!y.empty();)k=y.dequeue(),S=Bi(k.part,s,n),ji(m,b,S,k.part.id)||(I=Sf(k,d,n,s,f,_),w=xf(m,b,I),m.push({keypoints:I,score:w}));return[2,m]}})})}function Af(a){var e=a.shape,r=e[0],i=e[1],s=e[2];return Ye(function(){var t,u,p=Vt(a,[r*i,s]),h=pu(p,0),d=Tn(it(h,Wn(i,"int32")),1),n=Tn((t=h,u=i,Ye(function(){var f=it(t,Wn(u,"int32"));return mn(t,$e(f,Wn(u,"int32")))})),1);return gr([d,n],1)})}function Tf(a,e,r){return Ye(function(){var i=function(s,t){for(var u=[],p=0;p<St.length;p++){var h=s.get(p,0).valueOf(),d=s.get(p,1).valueOf(),n=Rf(h,d,p,t),f=n.x,_=n.y;u.push(_),u.push(f)}return rn(u,[St.length,2])}(a,r);return ot(Rn($e(a.toTensor(),Wn(e,"int32")),"float32"),i)})}function Rf(a,e,r,i){return{y:i.get(a,e,r),x:i.get(a,e,r+St.length)}}function Ff(a,e,r){return be(this,void 0,void 0,function(){var i,s,t,u,p,h,d,n,f,_;return _e(this,function(m){switch(m.label){case 0:return i=0,s=Af(a),[4,Promise.all([a.buffer(),e.buffer(),s.buffer()])];case 1:return t=m.sent(),u=t[0],p=t[1],h=t[2],[4,(d=Tf(h,r,p)).buffer()];case 2:return n=m.sent(),f=Array.from(function(y,b){for(var k=b.shape[0],S=new Float32Array(k),I=0;I<k;I++){var w=b.get(I,0),A=b.get(I,1);S[I]=y.get(w,A,I)}return S}(u,h)),_=f.map(function(y,b){return i+=y,{y:n.get(b,0),x:n.get(b,1),score:y,name:St[b]}}),s.dispose(),d.dispose(),[2,{keypoints:_,score:i/_.length}]}})})}function pa(a,e){return(a-1)%e==0}var da="https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/",ha="https://storage.googleapis.com/tfjs-models/savedmodel/posenet/resnet50/";function fa(a,e){return function(r,i){return(r-1)%i==0}(a,e)?a:Math.floor(a/e)*e+1}var ma=function(){function a(e,r){this.posenetModel=e;var i=this.posenetModel.inputs[0].shape;et(i[1]===-1&&i[2]===-1,function(){return"Input shape ["+i[1]+", "+i[2]+"] must both be equal to or -1"});var s,t,u=(s=r.inputResolution,t=r.outputStride,{height:fa(s.height,t),width:fa(s.width,t)});(function(p){et(yf.indexOf(p)>=0,function(){return"outputStride of "+p+" is invalid. It must be either 8 or 16."})})(r.outputStride),function(p,h){et(pa(p.height,h),function(){return"height of "+p.height+" is invalid for output stride "+h+"."}),et(pa(p.width,h),function(){return"width of "+p.width+" is invalid for output stride "+h+"."})}(u,r.outputStride),this.inputResolution=u,this.outputStride=r.outputStride,this.architecture=r.architecture}return a.prototype.estimatePoses=function(e,r){return r===void 0&&(r=bf),be(this,void 0,void 0,function(){var i,s,t,u,p,h,d,n,f,_,m,y,b,k,S;return _e(this,function(I){switch(I.label){case 0:return i=function(w){var A=w;if(A.maxPoses==null&&(A.maxPoses=1),A.maxPoses<=0)throw new Error("Invalid maxPoses "+A.maxPoses+". Should be > 0.");if(A.maxPoses>1){if((A=xe(xe({},_f),A)).scoreThreshold<0||A.scoreThreshold>1)throw new Error("Invalid scoreThreshold "+A.scoreThreshold+". Should be in range [0.0, 1.0]");if(A.nmsRadius<=0)throw new Error("Invalid nmsRadius "+A.nmsRadius+".")}return A}(r),e==null?[2,[]]:(this.maxPoses=i.maxPoses,s=kr(e,{outputTensorSize:this.inputResolution,keepAspectRatio:!0,borderMode:"replicate"}),t=s.imageTensor,u=s.padding,p=this.architecture==="ResNet50"?ot(t,vf):Ni(t,[-1,1]),h=this.posenetModel.predict(p),this.architecture==="ResNet50"?(d=Re(h[2],[0]),n=Re(h[3],[0]),f=Re(h[0],[0]),_=Re(h[1],[0])):(d=Re(h[0],[0]),n=Re(h[1],[0]),f=Re(h[2],[0]),_=Re(h[3],[0])),m=Tr(n),this.maxPoses!==1?[3,2]:[4,Ff(m,d,this.outputStride)]);case 1:return b=I.sent(),y=[b],[3,4];case 2:return[4,Mf(m,d,f,_,this.outputStride,this.maxPoses,i.scoreThreshold,i.nmsRadius)];case 3:y=I.sent(),I.label=4;case 4:return k=Nn(e),S=function(w,A,O,F){var P=A.height,V=A.width,D=P/(O.height*(1-F.top-F.bottom)),j=V/(O.width*(1-F.left-F.right)),G=-F.top*O.height,X=-F.left*O.width;if(j===1&&D===1&&G===0&&X===0)return w;for(var $=0,le=w;$<le.length;$++)for(var Q=0,ce=le[$].keypoints;Q<ce.length;Q++){var ke=ce[Q];ke.x=(ke.x+X)*j,ke.y=(ke.y+G)*D}return w}(y,k,this.inputResolution,u),i.flipHorizontal&&(S=function(w,A){for(var O=0,F=w;O<F.length;O++)for(var P=0,V=F[O].keypoints;P<V.length;P++){var D=V[P];D.x=A.width-1-D.x}return w}(S,k)),t.dispose(),p.dispose(),Ze(h),d.dispose(),n.dispose(),f.dispose(),_.dispose(),m.dispose(),[2,S]}})})},a.prototype.dispose=function(){this.posenetModel.dispose()},a.prototype.reset=function(){},a}();function Ef(a){return a===void 0&&(a=ta),be(this,void 0,void 0,function(){var e,r,i,s,t;return _e(this,function(u){switch(u.label){case 0:return(e=function(n){var f=n||ta;if(f.architecture==null&&(f.architecture="MobileNetV1"),na.indexOf(f.architecture)<0)throw new Error("Invalid architecture "+f.architecture+". Should be one of "+na);if(f.inputResolution==null&&(f.inputResolution={height:257,width:257}),f.outputStride==null&&(f.outputStride=16),ra[f.architecture].indexOf(f.outputStride)<0)throw new Error("Invalid outputStride "+f.outputStride+". Should be one of "+ra[f.architecture]+" for architecture "+f.architecture+".");if(f.multiplier==null&&(f.multiplier=1),aa[f.architecture].indexOf(f.multiplier)<0)throw new Error("Invalid multiplier "+f.multiplier+". Should be one of "+aa[f.architecture]+" for architecture "+f.architecture+".");if(f.quantBytes==null&&(f.quantBytes=4),sa.indexOf(f.quantBytes)<0)throw new Error("Invalid quantBytes "+f.quantBytes+". Should be one of "+sa+" for architecture "+f.architecture+".");if(f.architecture==="MobileNetV1"&&f.outputStride===32&&f.multiplier!==1)throw new Error("When using an output stride of 32, you must select 1 as the multiplier.");return f}(a)).architecture!=="ResNet50"?[3,2]:(p=e.outputStride,h=e.quantBytes,d="model-stride"+p+".json",r=h===4?ha+"float/"+d:ha+"quant"+h+"/"+d,[4,gn(e.modelUrl||r)]);case 1:return i=u.sent(),[2,new ma(i,e)];case 2:return s=function(n,f,_){var m={1:"100",.75:"075",.5:"050"},y="model-stride"+n+".json";return _===4?da+"float/"+m[f]+"/"+y:da+"quant"+_+"/"+m[f]+"/"+y}(e.outputStride,e.multiplier,e.quantBytes),[4,gn(e.modelUrl||s)];case 3:return t=u.sent(),[2,new ma(t,e)]}var p,h,d})})}function Nf(a,e){return be(this,void 0,void 0,function(){var r,i;return _e(this,function(s){switch(a){case It.PoseNet:return[2,Ef(e)];case It.BlazePose:if(i=void 0,(r=e)!=null){if(r.runtime==="tfjs")return[2,lf(e)];if(r.runtime==="mediapipe")return[2,Oh(e)];i=r.runtime}throw new Error("Expect modelConfig.runtime to be either 'tfjs' or 'mediapipe', but got "+i);case It.MoveNet:return[2,gf(e)];default:throw new Error(a+" is not a supported model name.")}})})}const Wi=a=>(qu("data-v-e1066a71"),a=a(),Ku(),a),Cf={class:"FaceLandmarksDetection"},Of={class:"box"},Pf=Wi(()=>tn("h3",null,"输入",-1)),Df=Wi(()=>tn("h3",null,"输出",-1)),Lf=Uu({__name:"PoseDetection",setup(a){const e=[["left_ear","left_eye","nose","right_eye","right_ear"],["left_wrist","left_elbow","left_shoulder","right_shoulder","right_elbow","right_wrist"],["left_shoulder","left_hip","left_knee","left_ankle"],["right_shoulder","right_hip","right_knee","right_ankle"],["left_hip","right_hip"]],r=Vn(!1),i=Vn(.5);let s;const t=Vn(),{videoConfig:u,videoButtonClick:p,newVideoRef:h}=Hu({videoProceed:y,videoRef:t}),d=Vn(),{getImageData:n,drawImage:f,fillArc:_,drawLine:m}=Wu({canvasRef:d});async function y(){if(!u.status)return;f(h.value);const k=n();(await s.estimatePoses(k,{scoreThreshold:i.value})).forEach(I=>{const w={};I.keypoints.forEach(A=>{A.score>i.value&&_(A.x,A.y,1,"#409eff"),w[A.name]=A}),e.forEach(A=>{const O=A.map(F=>{const{x:P,y:V,score:D}=w[F];return D>i.value?[P,V]:null}).filter(F=>F);O.length>0&&m({data:O,color:"#67c23a"})})}),window.requestAnimationFrame(y)}async function b(){const k=It.MoveNet;s=await Nf(k),r.value=!0}return zu(async()=>{Ui("wasm").then(()=>b())}),(k,S)=>(jr(),Br("div",Cf,[tn("div",Of,[tn("div",null,[Pf,tn("video",{id:"video",ref_key:"videoRef",ref:t,autoplay:""},null,512)]),tn("div",null,[Df,tn("canvas",{id:"canvasRef",ref_key:"canvasRef",ref:d},null,512)])]),r.value?(jr(),Br("button",{key:0,onClick:S[0]||(S[0]=(...I)=>rr(p)&&rr(p)(...I))},$u(rr(u).status?"关闭":"开启")+"摄像头 ",1)):Gu("",!0)]))}});const $f=Xu(Lf,[["__scopeId","data-v-e1066a71"]]);export{$f as default};
