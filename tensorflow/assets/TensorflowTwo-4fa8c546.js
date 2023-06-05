import{d as Jt,f as ke,g as tt,h as Xt,c as j,i as z,F as Ae,r as De,j as Yt,o as x,k as Zt,v as Qt,t as at,n as Mt}from"./index-7b9c9dd6.js";import{u as ea}from"./useVideo-bd9c37bb.js";import{o as v,a as S,c as b,b as Ne,E as D,A as ta,d as ae,m as ue,s as fe,B as aa,D as sa,e as ra,L as na,r as k,f as pe,S as ia,M as oa,T as be,g as st,h as ua,R as pa,i as ma,j as la,k as _e,l as wt,n as rt,p as ve,q as Oe,t as Ee,v as St,u as ca,w as da,x as _t,y as ha,z as K,C as ye,F as vt,G as Ot,H as fa,I as ya,J as ga,K as Na,N as se,O as ba,P as Ta,Q as wa,U as Sa,V as _a,W as va,X as Oa,Y as nt,Z as it,_ as Ea,$ as Ia,a0 as ka,a1 as Ke,a2 as Je,a3 as Xe,a4 as ot,a5 as Q,a6 as Et,a7 as Ye,a8 as Aa,a9 as It,aa as kt,ab as Da,ac as $a,ad as Ca,ae as za,af as La,ag as Pa,ah as Va,ai as Fa,aj as ja,ak as xa,al as Ra,am as ut,an as Ba,ao as At,ap as Ha,aq as qa,ar as Wa,as as Ua,at as Ga,au as Ka,av as Ja,aw as Xa,ax as Ya,ay as Za,az as Qa,aA as Ma,aB as es,aC as ts,aD as as,aE as ss,aF as rs,aG as ns,aH as is,aI as G,aJ as R,aK as Te,aL as os,aM as us}from"./index-c5cb974d.js";import{c as Ie,m as Y,s as Z,t as ze,o as oe,p as me,R as ps,r as Dt,a as le,b as $t,g as Ct,d as zt,e as ms,f as ls,h as cs,i as ds,S as hs,D as fs,j as q,k as ys,l as gs,n as Ns,q as bs,u as Ts,v as ws,w as Ss,x as _s,y as vs,z as Os,A as Es,B as Is,C as ks,E as As,F as Ds,G as $s,H as Cs,I as zs,J as Ls,K as Ps,L as Vs,M as Fs,N as js,O as xs,P as Rs,Q as Bs,T as Hs,U as qs,V as Ws,W as Us,X as Gs,Y as Ks,Z as Js,_ as Xs,$ as Ys,a0 as Zs,a1 as Qs,a2 as Ms,a3 as er,a4 as tr,a5 as ar,a6 as sr,a7 as rr,a8 as nr,a9 as ir,aa as or,ab as ur,ac as pr,ad as mr,ae as lr,af as cr,ag as dr,ah as hr,ai as fr,aj as yr,ak as Le,al as gr,am as Nr,an as br,ao as Tr,ap as wr,aq as Sr,ar as _r,as as vr,at as Or,au as Er,av as Ir,aw as kr,ax as Ar,ay as Dr,az as $r,aA as Cr,aB as zr,aC as Lr,aD as Pr,aE as Vr,aF as Fr,aG as jr,aH as xr,aI as Rr,aJ as Br,aK as Hr,aL as qr,aM as Wr,aN as Ur,aO as Gr,aP as Lt,aQ as Kr,aR as Jr,aS as Xr,aT as Yr,aU as Zr,aV as Qr,aW as Mr,aX as en,aY as tn,aZ as an,a_ as sn,a$ as rn,b0 as nn,b1 as on,b2 as un,b3 as pn,b4 as mn,b5 as ln,b6 as cn,b7 as dn,b8 as hn,b9 as fn,ba as yn,bb as gn,bc as Nn,bd as bn,be as Tn,bf as re,bg as wn,bh as Sn,bi as _n,bj as Pt,bk as vn,bl as On,bm as En,bn as In,bo as kn,bp as An,bq as Dn,br as ce,bs as $n,bt as Cn,bu as Vt}from"./register_all_kernels-49f39960.js";import{_ as zn}from"./_plugin-vue_export-helper-c27b6911.js";/**
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
 */function Ln(a){S(Array.isArray(a),()=>"The argument passed to tf.addN() must be a list of tensors"),S(a.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${a.length}`);const e=a.map((r,i)=>b(r,`tensors${i}`,"addN")),t=e[0];e.forEach(r=>{if(r.dtype!==t.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),e.forEach(r=>{if(!Ne(r.shape,t.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const s=e;return D.runKernel(ta,s)}const Pn=v({addN_:Ln});/**
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
 */function Vn(a,e,t,s,r,i){const o=b(a,"forgetBias","basicLSTMCell"),u=b(e,"lstmKernel","basicLSTMCell"),p=b(t,"lstmBias","basicLSTMCell"),m=b(s,"data","basicLSTMCell"),l=b(r,"c","basicLSTMCell"),c=b(i,"h","basicLSTMCell"),d=Ie([m,c],1),f=Y(d,u),g=ae(f,p),y=g.shape[0],h=g.shape[1]/4,N=[y,h],_=Z(g,[0,0],N),A=Z(g,[0,h],N),T=Z(g,[0,h*2],N),I=Z(g,[0,h*3],N),$=ae(ue(fe(_),ze(A)),ue(l,fe(ae(o,T)))),w=ue(ze($),fe(I));return[$,w]}const Fn=v({basicLSTMCell_:Vn});/**
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
 */function jn(a,e){const t=b(a,"s0","broadcastArgs","int32"),s=b(e,"s1","broadcastArgs","int32");if(t.rank!==1)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${t.rank}`);if(s.rank!==1)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${s.rank}`);const r={s0:t,s1:s};return D.runKernel(aa,r)}const xn=v({broadcastArgs_:jn});/**
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
 */function Rn(a){const t={x:b(a,"x","diag")};return D.runKernel(sa,t)}const Bn=v({diag_:Rn});/**
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
 */function Hn(a,...e){const t=e.map((r,i)=>b(r,`tensors${i}`,"einsum")),s={equation:a};return D.runKernel(ra,t,s)}const qn=v({einsum_:Hn});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function Wn(a,e,t){if(t<=0)throw new Error("The number of values should be positive.");const s={start:a,stop:e,num:t};return D.runKernel(na,{},s)}/**
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
 */const de=2147483648;function Un(a,e,t="left"){const s=b(a,"sortedSequence","searchSorted"),r=b(e,"values","searchSorted"),i=s.shape[s.shape.length-1],o=r.shape[r.shape.length-1],u=k(s,[-1,i]),p=k(r,[-1,o]);if(u.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(u.shape[0]!==p.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(pe(p.shape)>=de)throw new Error(`values tensor size must less than ${de}`);if(u.shape[1]>=de)throw new Error(`trailing dim_size must less than ${de} for int32 output type, was ${u.shape[1]}`);const m={sortedSequence:u,values:p},l={side:t};return D.runKernel(ia,m,l)}const Ze=v({searchSorted_:Un});/**
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
 */function Gn(a,e){return Ze(a,e,"left")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function Kn(a,e,t,s,r=!1){const o={x:b(a,"x","maxPoolWithArgmax")},u={filterSize:e,strides:t,pad:s,includeBatchInIndex:r},p=D.runKernel(oa,o,u);return{result:p[0],indexes:p[1]}}const Jn=v({maxPoolWithArgmax_:Kn});/**
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
 */function Xn(a,e,{indexing:t="xy"}={}){if(t!=="xy"&&t!=="ij")throw new TypeError(`${t} is not a valid third argument to meshgrid`);if(a===void 0)return[];let s=b(a,"x","meshgrid",a instanceof be?a.dtype:"float32");if(e===void 0)return[s];let r=b(e,"y","meshgrid",e instanceof be?e.dtype:"float32");const i=pe(s.shape),o=pe(r.shape);return t==="xy"?(s=k(s,[1,-1]),r=k(r,[-1,1]),[Y(oe([o,1],s.dtype),s),Y(r,oe([1,i],r.dtype))]):(s=k(s,[-1,1]),r=k(r,[1,-1]),[Y(s,oe([1,o],s.dtype)),Y(oe([i,1],r.dtype),r)])}function Yn(a,e,t,s){const r=b(e,"data","multiRNNCell"),i=st(t,"c","multiRNNCell"),o=st(s,"h","multiRNNCell");let u=r;const p=[];for(let c=0;c<a.length;c++){const d=a[c](u,i[c],o[c]);p.push(d[0]),p.push(d[1]),u=d[1]}const m=[],l=[];for(let c=0;c<p.length;c+=2)m.push(p[c]),l.push(p[c+1]);return[m,l]}const Zn=v({multiRNNCell_:Yn});/**
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
 */function Qn(a,e,t,s=!1){const r=b(a,"logits","multinomial"),i=r.size,o=r.rank;if(i<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${i}.`);if(o>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${o}`);t=t||Math.random();const p={logits:o===1?k(r,[1,-1]):r},m={numSamples:e,seed:t,normalized:s},l=D.runKernel(ua,p,m);return o===1?k(l,[l.size]):l}const Mn=v({multinomial_:Qn});function ei(a,e){const t=b(a,"v1","outerProduct"),s=b(e,"v2","outerProduct");S(t.rank===1&&s.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${t.rank} and ${s.rank}.`);const r=k(t,[-1,1]),i=k(s,[1,-1]);return Y(r,i)}const ti=v({outerProduct_:ei});function ai(a,e,t=0){return S(e.length===2,()=>"Invalid number of paddings. Must be length of 2."),me(a,[e],t)}const si=v({pad1d_:ai});function ri(a,e,t=0){return S(e.length===2&&e[0].length===2&&e[1].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),me(a,e,t)}const ni=v({pad2d_:ri});function ii(a,e,t=0){return S(e.length===3&&e[0].length===2&&e[1].length===2&&e[2].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),me(a,e,t)}const oi=v({pad3d_:ii});function ui(a,e,t=0){return S(e.length===4&&e[0].length===2&&e[1].length===2&&e[2].length===2&&e[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),me(a,e,t)}const pi=v({pad4d_:ui});/**
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
 */function mi(a,e,t,s){const r=a.map((l,c)=>b(l,`tensors${c}`,"raggedGather","int32")),i=b(e,"paramsDenseValues","raggedGather"),o=b(t,"indices","raggedGather","int32"),u={paramsNestedSplits:r,paramsDenseValues:i,indices:o},p={outputRaggedRank:s},m=D.runKernel(pa,u,p);return{outputNestedSplits:m.slice(0,m.length-1),outputDenseValues:m[m.length-1]}}const li=v({raggedGather_:mi});/**
 * @license
 * Copyright 2022 Google LLC.
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
 */function ci(a,e,t){const s=b(a,"starts","raggedRange"),r=b(e,"limits","raggedRange",s.dtype),i=b(t,"deltas","raggedRange",s.dtype),o={starts:s,limits:r,deltas:i},u=D.runKernel(ma,o);return{rtNestedSplits:u[0],rtDenseValues:u[1]}}const di=v({raggedRange_:ci});/**
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
 */function hi(a,e,t,s,r){const i=b(a,"shape","raggedTensorToTensor","int32"),o=b(e,"values","raggedTensorToTensor"),u=b(t,"defaultValue","raggedTensorToTensor",o.dtype),p=s.map((c,d)=>b(c,`tensors${d}`,"raggedTensorToTensor","int32")),m={shape:i,values:o,defaultValue:u,rowPartitionTensors:p},l={rowPartitionTypes:r};return D.runKernel(la,m,l)}const fi=v({raggedTensorToTensor_:hi});/**
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
 */function yi(a,e,t){_e(a);const s=pe(a);let r=null;if(t==null||t==="float32")r=new Float32Array(s);else if(t==="int32")r=new Int32Array(s);else if(t==="bool")r=new Uint8Array(s);else throw new Error(`Unknown data type ${t}`);for(let i=0;i<s;i++)r[i]=e();return D.makeTensor(r,a,t)}const gi=v({rand_:yi});/**
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
 */function Ni(a,e,t=1,s="float32",r){if(_e(a),t==null&&(t=1),s==null&&(s="float32"),s!=="float32"&&s!=="int32")throw new Error(`Unsupported data type ${s}`);const i=new ps(e,t,s,r),o=wt(a,s);for(let u=0;u<o.values.length;u++)o.values[u]=i.nextValue();return o.toTensor()}const bi=v({randomGamma_:Ni});/**
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
 */function Ti(a,e,t){if(e!=null&&e==="bool")throw new Error(`Unsupported data type ${e}`);return Dt(a,0,1,e,t)}const wi=v({randomStandardNormal_:Ti});/**
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
 */function Si(a){const e=b(a,"x","reverse");return S(e.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${e.rank}.`),le(e,0)}const _i=v({reverse1d_:Si});/**
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
 */function vi(a,e){const t=b(a,"x","reverse");return S(t.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${t.rank}.`),le(t,e)}const Oi=v({reverse2d_:vi});/**
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
 */function Ei(a,e){const t=b(a,"x","reverse");return S(t.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${t.rank}.`),le(t,e)}const Ii=v({reverse3d_:Ei});/**
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
 */function ki(a,e){const t=b(a,"x","reverse");return S(t.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${t.rank}.`),le(t,e)}const Ai=v({reverse4d_:ki});/**
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
 */async function Di(a,e){const t=b(a,"x","setdiff1d"),s=b(e,"y","setdiff1d");S(t.dtype===s.dtype,()=>`x and y should have the same dtype, but got x (${t.dtype}) and y (${s.dtype}).`),S(t.rank===1,()=>`x should be 1D tensor, but got x (${t.shape}).`),S(s.rank===1,()=>`y should be 1D tensor, but got y (${s.shape}).`);const r=await t.data(),i=await s.data(),o=new Set(i);let u=0;for(let l=0;l<r.length;l++)o.has(r[l])||u++;const p=new rt([u],t.dtype),m=new rt([u],"int32");for(let l=0,c=0;l<r.length;l++)o.has(r[l])||(p.values[c]=r[l],m.values[c]=l,c++);return[p.toTensor(),m.toTensor()]}const $i=Di;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function Ft(a,e,t){if(ve(a),e!=null&&e.length!==3)throw new Error("tensor3d() requires shape to have three numbers");const s=Oe(a,t);if(s.length!==3&&s.length!==1)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return Ee(a,e,s,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function Ci(a,e,t){if(ve(a),e!=null&&e.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const s=Oe(a,t);if(s.length!==4&&s.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return Ee(a,e,s,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function zi(a,e,t){if(ve(a),e!=null&&e.length!==5)throw new Error("tensor5d() requires shape to have five numbers");const s=Oe(a,t);if(s.length!==5&&s.length!==1)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return Ee(a,e,s,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function Li(a,e,t){if(ve(a),e!=null&&e.length!==6)throw new Error("tensor6d() requires shape to have six numbers");const s=Oe(a,t);if(s.length!==6&&s.length!==1)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return e=e||s,Ee(a,e,s,t)}/**
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
 */function Pi(a,e,t){const s=b(a,"tensor","tensorScatterupdate"),r=b(e,"indices","tensorScatterupdate","int32"),i=b(t,"updates","tensorScatterupdate");if(St(i,r,s.shape),s.dtype!==i.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${s.dtype} and ${i.dtype}.`);const o={tensor:s,indices:r,updates:i},u={};return D.runKernel(ca,o,u)}const Vi=v({tensorScatterUpdate_:Pi});/**
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
 */function Fi(a,e){return Ze(a,e,"right")}/**
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
 */async function ji(a){const e=b(a,"condition","whereAsync","bool"),t=await e.data(),s=da(e.shape,t);return a!==e&&e.dispose(),s}const jt=ji;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */async function xi(a,e,t){const s=b(a,"tensor","boolMask"),r=b(e,"mask","boolMask","bool"),i=t??0,o=r.rank,u=s.shape;S(o>0,()=>"mask cannot be scalar"),_t(u.slice(i,i+o),r.shape,"mask's shape must match the first K dimensions of tensor's shape,");let p=1;for(let y=i;y<i+o;y++)p*=u[y];const m=u.slice(0,i).concat([p],u.slice(i+o)),l=k(s,m),c=k(r,[-1]),d=await jt(c),f=$t(d,[1]),g=Ct(l,f,i);return a!==s&&s.dispose(),e!==r&&r.dispose(),f.dispose(),l.dispose(),c.dispose(),d.dispose(),g}const Ri=xi;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function Bi(a,e,t,s,r=!0){const i=b(a,"v","movingAverage"),o=b(e,"x","movingAverage"),u=b(t,"decay","movingAverage");ha(i,o),S(Ne(i.shape,o.shape),()=>"Shape mismatch in v and x");const p=K(1),m=ye(p,u);let l=ue(ye(o,i),m);if(r){S(s!=null,()=>"When using zeroDebias: true, step is required.");const c=b(s,"step","movingAverage");l=vt(l,ye(p,Ot(u,c)))}return ae(i,l)}const Hi=v({movingAverage_:Bi});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function qi(a,e,t){_e(t);const s=b(a,"indices","scatterND","int32"),r=b(e,"updates","scatterND");St(r,s,t);const i={indices:s,updates:r},o={shape:t};return D.runKernel(fa,i,o)}const Wi=v({scatterND_:qi});function Ui(a,e,t,s){if(a.dtype!=="int32")throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${a.dtype}.`);if(a.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${a.shape}.`);const r=a.rank>0?a.shape[0]:1,i=a.rank>1?a.shape[1]:1;if(t.length!==i)throw new Error(`outputShape has incorrect number of elements:, ${t.length}, should be: ${i}.`);const o=e.size;if(!(e.rank===0||e.rank===1&&o===r))throw new Error(`sparseValues has incorrect shape ${e.shape}, should be [] or [${r}]`);if(e.dtype!==s.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function Gi(a,e,t,s=0){_e(t);const r=b(a,"sparseIndices","sparseToDense","int32"),i=b(e,"sparseValues","sparseToDense","string_or_numeric"),o=b(s,"defaultValue","sparseToDense",i.dtype);Ui(r,i,t,o);const u={sparseIndices:r,sparseValues:i,defaultValue:o},p={outputShape:t};return D.runKernel(ya,u,p)}const Ki=v({sparseToDense_:Gi});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function Ji(a,e){const t=b(e,"indices","gatherND","int32"),r={params:b(a,"x","gatherND","string_or_numeric"),indices:t};return D.runKernel(ga,r)}const Xi=v({gatherND_:Ji});/**
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
 */async function Yi(a,e,t=1){const s=b(a,"predictions","inTopK"),r=b(e,"targets","inTopK");S(s.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${s.rank}`),S(s.rank-1===r.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${s.rank} and targets rank ${r.rank}`),_t(s.shape.slice(0,s.shape.length-1),r.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const i=s.shape[s.shape.length-1];S(t>0&&t<=i,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${i}), but got ${t}`);const o=await s.data(),u=await r.data(),[p,m]=[o.length/i,i],l=Na("bool",p);for(let c=0;c<p;c++){const d=c*m,f=o.subarray(d,d+m),g=[];for(let y=0;y<f.length;y++)g.push({value:f[y],index:y});g.sort((y,h)=>h.value-y.value),l[c]=0;for(let y=0;y<t;y++)if(g[y].index===u[c]){l[c]=1;break}}return a!==s&&s.dispose(),e!==r&&r.dispose(),se(l,r.shape,"bool")}const Zi=Yi;/**
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
 */function Qi({x:a,filter:e,strides:t,pad:s,dataFormat:r="NHWC",dilations:i=[1,1],dimRoundingMode:o,bias:u,activation:p="linear",preluActivationWeights:m,leakyreluAlpha:l}){if(ba(D.state.gradientDepth,p)===!1){let I=zt(a,e,t,s,r,i,o);return u!=null&&(I=ae(I,u)),Ta(I,p,m,l)}const c=b(a,"x","depthwiseConv2d","float32"),d=b(e,"filter","depthwiseConv2d","float32");let f=c,g=!1;c.rank===3&&(g=!0,f=k(c,[1,c.shape[0],c.shape[1],c.shape[2]])),S(f.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${f.rank}.`),S(d.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${d.rank}.`),S(f.shape[3]===d.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${f.shape[3]}) must match the inChannels dimension in filter ${d.shape[2]}.`),i==null&&(i=[1,1]),S(wa(t,i),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${t} and dilations '${i}'`),Sa("fused depthwiseConv2d",s,o);const y=_a(f.shape,d.shape,t,i,s,o,!0);let h;u!=null&&(h=b(u,"bias","fused conv2d"),[h]=va(h,c),Oa(y.outShape,h.shape));let N;m!=null&&(N=b(m,"prelu weights","fused depthwiseConv2d"));const _=(I,$)=>{S(Ea(i),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${i}'`);const[w,C,O,E]=$,V=Ia(I,O,p),H=ms(C.shape,V,w,t,s,i,o),ne=ls(C,V,w.shape,t,s,i,o);if(E!=null){const te=ka(h,V);return[H,ne,te]}return[H,ne]},A={x:f,filter:d,bias:h,preluActivationWeights:N},T={strides:t,pad:s,dataFormat:r,dilations:i,dimRoundingMode:o,activation:p,leakyreluAlpha:l};return u==null?nt(($,w,C)=>{let O=D.runKernel(it,A,T);return C([w,$,O]),g&&(O=k(O,[O.shape[1],O.shape[2],O.shape[3]])),{value:O,gradFunc:_}})(f,d):nt(($,w,C,O)=>{let E=D.runKernel(it,A,T);return O([w,$,E,C]),g&&(E=k(E,[E.shape[1],E.shape[2],E.shape[3]])),{value:E,gradFunc:_}})(f,d,h)}const Mi=v({fusedDepthwiseConv2d_:Qi});/**
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
 */const eo=Object.freeze(Object.defineProperty({__proto__:null,conv2d:cs,depthwiseConv2d:Mi,matMul:ds},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const to="model",ao=".json",so=".weights.bin";function pt(a){return new Promise(e=>setTimeout(e)).then(a)}class M{constructor(e){if(!Q().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");e.startsWith(M.URL_SCHEME)&&(e=e.slice(M.URL_SCHEME.length)),(e==null||e.length===0)&&(e=to),this.modelJsonFileName=e+ao,this.weightDataFileName=e+so}async save(e){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const t=window.URL.createObjectURL(new Blob([e.weightData],{type:"application/octet-stream"}));if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const s=[{paths:["./"+this.weightDataFileName],weights:e.weightSpecs}],r=Et(e,s),i=window.URL.createObjectURL(new Blob([JSON.stringify(r)],{type:"application/json"})),o=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(o.download=this.modelJsonFileName,o.href=i,await pt(()=>o.dispatchEvent(new MouseEvent("click"))),e.weightData!=null){const u=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;u.download=this.weightDataFileName,u.href=t,await pt(()=>u.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:Ye(e)}}}}M.URL_SCHEME="downloads://";class ro{constructor(e){if(e==null||e.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,t)=>{const s=new FileReader;s.onload=r=>{const i=JSON.parse(r.target.result),o=i.modelTopology;if(o==null){t(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(i.weightsManifest==null){t(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){e({modelTopology:o});return}const p=Je(i,m=>this.loadWeights(m));e(p)},s.onerror=r=>t(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),s.readAsText(this.jsonFile)})}loadWeights(e){const t=[],s=[];for(const o of e)t.push(...o.weights),s.push(...o.paths);const r=this.checkManifestAndWeightFiles(e),i=s.map(o=>this.loadWeightsFile(o,r[o]));return Promise.all(i).then(o=>[t,Xe(o)])}loadWeightsFile(e,t){return new Promise((s,r)=>{const i=new FileReader;i.onload=o=>{const u=o.target.result;s(u)},i.onerror=o=>r(`Failed to weights data from file of path '${e}'.`),i.readAsArrayBuffer(t)})}checkManifestAndWeightFiles(e){const t=[],s=this.weightsFiles.map(i=>ot(i.name)),r={};for(const i of e)i.paths.forEach(o=>{const u=ot(o);if(t.indexOf(u)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${u}'`);if(t.push(u),s.indexOf(u)===-1)throw new Error(`Weight file with basename '${u}' is not provided.`);r[o]=this.weightsFiles[s.indexOf(u)]});if(t.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${t.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}const no=a=>Q().getBool("IS_BROWSER")&&!Array.isArray(a)&&a.startsWith(M.URL_SCHEME)?io(a.slice(M.URL_SCHEME.length)):null;Ke.registerSaveRouter(no);function io(a="model"){return new M(a)}function oo(a){return new ro(a)}/**
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
 */function mt(a,e,t,s){o(a),t=t??0,s=s??1,u(t,s);let r=0;const i=p=>(p.then(m=>{const l=t+ ++r/a.length*(s-t);return e(l),m}),p);function o(p){S(p!=null&&Array.isArray(p)&&p.length>0,()=>"promises must be a none empty array")}function u(p,m){S(p>=0&&p<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${p}`),S(m>=0&&m<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${m}`),S(m>=p,()=>`startFraction must be no more than endFraction, but got startFraction ${p} and endFraction ${m}`)}return Promise.all(a.map(i))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */async function xt(a,e){e==null&&(e={});const t=e.fetchFunc==null?Q().platform.fetch:e.fetchFunc,s=a.map(c=>t(c,e.requestInit,{isBinary:!0})),r=0,i=.5,u=(e.onProgress==null?await Promise.all(s):await mt(s,e.onProgress,r,i)).map(c=>c.arrayBuffer()),p=.5,m=1;return e.onProgress==null?await Promise.all(u):await mt(u,e.onProgress,p,m)}async function uo(a,e="",t,s){return Rt(o=>xt(o,{requestInit:s}))(a,e,t)}function Rt(a){return async(e,t="",s)=>{const r=e.map(()=>!1),i={},o=s!=null?s.map(()=>!1):[],u=[];if(e.forEach((f,g)=>{let y=0;f.weights.forEach(h=>{const N="quantization"in h?h.quantization.dtype:h.dtype,_=Aa[N]*pe(h.shape),A=()=>{r[g]=!0,i[g]==null&&(i[g]=[]),i[g].push({manifestEntry:h,groupOffset:y,sizeBytes:_})};s!=null?s.forEach((T,I)=>{T===h.name&&(A(),o[I]=!0)}):A(),u.push(h.name),y+=_})}),!o.every(f=>f)){const f=s.filter((g,y)=>!o[y]);throw new Error(`Could not find weights in manifest with names: ${f.join(", ")}. 
Manifest JSON has weights with names: ${u.join(", ")}.`)}const p=r.reduce((f,g,y)=>(g&&f.push(y),f),[]),m=[];p.forEach(f=>{e[f].paths.forEach(g=>{const y=t+(t.endsWith("/")?"":"/")+g;m.push(y)})});const l=await a(m),c={};let d=0;return p.forEach(f=>{const g=e[f].paths.length;let y=0;for(let T=0;T<g;T++)y+=l[d+T].byteLength;const h=new ArrayBuffer(y),N=new Uint8Array(h);let _=0;for(let T=0;T<g;T++){const I=new Uint8Array(l[d+T]);N.set(I,_),_+=I.byteLength}i[f].forEach(T=>{const I=h.slice(T.groupOffset,T.groupOffset+T.sizeBytes),$=It(I,[T.manifestEntry]);for(const w in $)c[w]=$[w]}),d+=g}),c}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const po="application/octet-stream",mo="application/json";class Qe{constructor(e,t){if(this.DEFAULT_METHOD="POST",t==null&&(t={}),this.weightPathPrefix=t.weightPathPrefix,this.onProgress=t.onProgress,this.weightUrlConverter=t.weightUrlConverter,t.fetchFunc!=null?(S(typeof t.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=t.fetchFunc):this.fetch=Q().platform.fetch,S(e!=null&&e.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(e)&&S(e.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${e.length}).`),this.path=e,t.requestInit!=null&&t.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=t.requestInit||{}}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const t=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);t.body=new FormData;const s=[{paths:["./model.weights.bin"],weights:e.weightSpecs}],r=Et(e,s);t.body.append("model.json",new Blob([JSON.stringify(r)],{type:mo}),"model.json"),e.weightData!=null&&t.body.append("model.weights.bin",new Blob([e.weightData],{type:po}),"model.weights.bin");const i=await this.fetch(this.path,t);if(i.ok)return{modelArtifactsInfo:Ye(e),responses:[i]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${i.status}.`)}async load(){const e=await this.fetch(this.path,this.requestInit);if(!e.ok)throw new Error(`Request to ${this.path} failed with status code ${e.status}. Please verify this URL points to the model JSON of the model to load.`);let t;try{t=await e.json()}catch{let o=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?o+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":o+=" Please make sure the server is serving valid JSON for this request.",new Error(o)}const s=t.modelTopology,r=t.weightsManifest;if(s==null&&r==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return Je(t,i=>this.loadWeights(i))}async loadWeights(e){const t=Array.isArray(this.path)?this.path[1]:this.path,[s,r]=lo(t),i=this.weightPathPrefix||s,o=kt(e),u=[],p=[];for(const l of e)for(const c of l.paths)this.weightUrlConverter!=null?p.push(this.weightUrlConverter(c)):u.push(i+c+r);this.weightUrlConverter&&u.push(...await Promise.all(p));const m=await xt(u,{requestInit:this.requestInit,fetchFunc:this.fetch,onProgress:this.onProgress});return[o,Xe(m)]}}Qe.URL_SCHEME_REGEX=/^https?:\/\//;function lo(a){const e=a.lastIndexOf("/"),t=a.lastIndexOf("?"),s=a.substring(0,e),r=t>e?a.substring(t):"";return[s+"/",r]}function Pe(a){return a.match(Qe.URL_SCHEME_REGEX)!=null}const Bt=(a,e)=>{if(typeof fetch>"u"&&(e==null||e.fetchFunc==null))return null;{let t=!0;if(Array.isArray(a)?t=a.every(s=>Pe(s)):t=Pe(a),t)return Me(a,e)}return null};Ke.registerSaveRouter(Bt);Ke.registerLoadRouter(Bt);function Me(a,e){return new Qe(a,e)}function co(a,e){return Me(a,e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */class $e{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}}class Ht{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}}class ho{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=t=>Promise.resolve(e.save(t)))}}function fo(a,e,t,s){const r=arguments;return new ho(qt(...r))}function qt(a,e,t,s){return arguments.length===1?a.modelTopology!=null||a.weightSpecs!=null?new $e(a):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new $e({modelTopology:a})):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new $e({modelTopology:a,weightSpecs:e,weightData:t,trainingConfig:s}))}function yo(a){return new Ht(a)}function go(a){return new Ht(a)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const Wt=Object.freeze(Object.defineProperty({__proto__:null,browserFiles:oo,browserHTTPRequest:co,concatenateArrayBuffers:Xe,copyModel:Da,decodeWeights:It,encodeWeights:$a,fromMemory:fo,fromMemorySync:qt,getLoadHandlers:Ca,getModelArtifactsForJSON:Je,getModelArtifactsForJSONSync:za,getModelArtifactsInfoForJSON:Ye,getSaveHandlers:La,getWeightSpecs:kt,http:Me,isHTTPScheme:Pe,listModels:Pa,loadWeights:uo,moveModel:Va,registerLoadRouter:Fa,registerSaveRouter:ja,removeModel:xa,weightsLoaderFactory:Rt,withSaveHandler:yo,withSaveHandlerSync:go},Symbol.toStringTag,{value:"Module"}));/**
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
 */let J;function No(a,e=3){if(e>4)throw new Error("Cannot construct Tensor with more than 4 channels from pixels.");if(a==null)throw new Error("pixels passed to tf.browser.fromPixels() can not be null");let t=!1,s=!1,r=!1,i=!1,o=!1,u=!1;if(a.data instanceof Uint8Array)t=!0;else if(typeof ImageData<"u"&&a instanceof ImageData)s=!0;else if(typeof HTMLVideoElement<"u"&&a instanceof HTMLVideoElement)r=!0;else if(typeof HTMLImageElement<"u"&&a instanceof HTMLImageElement)i=!0;else if(a.getContext!=null)o=!0;else if(typeof ImageBitmap<"u"&&a instanceof ImageBitmap)u=!0;else throw new Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${a.constructor.name}`);if(Ra(ut,D.backendName)!=null){const g={pixels:a},y={numChannels:e};return D.runKernel(ut,g,y)}const[m,l]=r?[a.videoWidth,a.videoHeight]:[a.width,a.height];let c;if(o)c=a.getContext("2d").getImageData(0,0,m,l).data;else if(s||t)c=a.data;else if(i||r||u){if(J==null)if(typeof document>"u")if(typeof OffscreenCanvas<"u"&&typeof OffscreenCanvasRenderingContext2D<"u")J=new OffscreenCanvas(1,1).getContext("2d");else throw new Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.");else J=document.createElement("canvas").getContext("2d",{willReadFrequently:!0});J.canvas.width=m,J.canvas.height=l,J.drawImage(a,0,0,m,l),c=J.getImageData(0,0,m,l).data}let d;if(e===4)d=new Int32Array(c);else{const g=m*l;d=new Int32Array(g*e);for(let y=0;y<g;y++)for(let h=0;h<e;++h)d[y*e+h]=c[y*4+h]}return Ft(d,[l,m,e],"int32")}const lt=v({fromPixels_:No});/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function bo(a){return new hs(a)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function ct(a){return new fs(a)}/**
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
 */const To={};function Ut(a){return To[a]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function n(a,e,t,s,r){const i=e.inputParams[a];if(i&&i.inputIndexStart!==void 0){const u=i.inputIndexStart,p=i.inputIndexEnd===0?void 0:i.inputIndexEnd===void 0?u+1:i.inputIndexEnd,m=u<0?e.inputNames.length+u:u;if(i.type==="tensor")return L(e.inputNames[m],t,s,r);if(i.type==="tensors"){const d=e.inputs.slice(u,p);return e.inputNames.slice(u,p).filter((g,y)=>{var h;return((h=d[y])===null||h===void 0?void 0:h.op)!=="NoOp"}).map(g=>L(g,t,s,r))}const l=L(e.inputNames[m],t,s,r),c=l.dataSync();return i.type==="number"?c[0]:Ba(l.shape,c)}const o=e.attrParams[a];return o&&o.value}function L(a,e,t,s){const[r,i]=F(a,t);if(s!=null){const u=s.getHashTableHandleByName(r);if(u!=null)return u}const o=t.currentContextIds.find(u=>!!e[we(r,u)]);return o!==void 0?e[we(r,o)][i]:void 0}function dt(a,e,t){return e[we(a,t.currentContextId)]}function W(a,e){const[t,s,r]=F(a,e);return[we(t,e&&e.currentContextId),s,r]}function we(a,e){return e?`${a}-${e}`:a}function F(a,e){if(a==="")return["",0,void 0];const t=e!=null&&e.parseNodeNameCache!=null;if(t){const i=e.parseNodeNameCache.get(a);if(i!=null)return i}const s=a.split(":");let r;if(s.length===1)r=[a,0,void 0];else{const i=s[0],o=s.length===3?s[1]:void 0,u=Number(s[s.length-1]);r=[i,u,o]}return t&&e.parseNodeNameCache.set(a,r),r}function ge(a,e,t){let s=n("pad",a,e,t);if(s==="explicit"){s=n("explicitPaddings",a,e,t);const r=[[0,0],[0,0],[0,0],[0,0]];for(let i=0;i<4;i++)r[i][0]=s[i*2],r[i][1]=s[i*2+1];return r}return s}function U(a){return a.kept?a:At(a)}/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const wo=[{tfOpName:"Add",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddV2",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddN",category:"arithmetic",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"BiasAdd",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"Sub",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"RealDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Div",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"DivNoNan",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mul",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Maximum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Minimum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Pow",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SquaredDifference",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorMod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],So=Object.freeze(Object.defineProperty({__proto__:null,json:wo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const _o=[{tfOpName:"Abs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan2",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Ceil",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ClipByValue",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"clipValueMin",type:"number"},{start:2,name:"clipValueMax",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Complex",category:"basic_math",inputs:[{start:0,name:"real",type:"tensor"},{start:1,name:"imag",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ComplexAbs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Elu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Exp",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Floor",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Imag",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Neg",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Real",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Prelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"alpha",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu6",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Selu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sigmoid",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Rsqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Square",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sign",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Round",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Expm1",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log1p",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Reciprocal",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Softplus",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Erf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Prod",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axes",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LeakyRelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"alpha",name:"alpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsNan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsFinite",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsInf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],vo=Object.freeze(Object.defineProperty({__proto__:null,json:_o},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Oo=[{tfOpName:"EmptyTensorList",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"maxNumElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"LoopCond",category:"control",inputs:[{start:0,name:"pred",type:"tensor"}]},{tfOpName:"Switch",category:"control",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"pred",type:"tensor"}]},{tfOpName:"Merge",category:"control",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"Enter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"frame_name",name:"frameName",type:"string"},{tfName:"is_constant",name:"isConstant",type:"bool"}]},{tfOpName:"Exit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NextIteration",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayV3",category:"control",inputs:[{start:0,name:"size",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"dynamic_size",name:"dynamicSize",type:"bool"},{tfName:"clear_after_read",name:"clearAfterRead",type:"bool"},{tfName:"identical_element_shapes",name:"identicalElementShapes",type:"bool"},{tfName:"tensor_array_name",name:"name",type:"string"}]},{tfOpName:"TensorArrayWriteV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayReadV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayGatherV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"}]},{tfOpName:"TensorArrayScatterV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArrayConcatV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape_except0",name:"elementShapeExcept0",type:"shape",notSupported:!0}]},{tfOpName:"TensorArraySplitV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"tensor",type:"tensor"},{start:2,name:"lengths",type:"number[]"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArraySizeV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}]},{tfOpName:"TensorArrayCloseV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"}]},{tfOpName:"StatelessIf",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"If",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"StatelessWhile",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"While",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"TensorListScatter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListScatterV2",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"},{start:3,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGather",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListSetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListReserve",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListFromTensor",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListStack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"},{tfName:"num_elements",name:"numElements",type:"dtype"}]},{tfOpName:"TensorListSplit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"},{start:2,name:"lengths",type:"number[]"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcat",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcatV2",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPopBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPushBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListLength",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}]},{tfOpName:"TensorListResize",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"size",type:"number"}]}],Eo=Object.freeze(Object.defineProperty({__proto__:null,json:Oo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Io=[{tfOpName:"AvgPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[],notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPoolWithArgmax",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"include_batch_in_index",name:"includeBatchInIndex",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AvgPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Conv1D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"stride",name:"stride",type:"number"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NWC"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"dilation",name:"dilation",type:"number",defaultValue:1}]},{tfOpName:"Conv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"useCudnnOnGpu",name:"useCudnnOnGpu",type:"bool"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"_FusedConv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"use_cudnn_on_gpu",name:"useCudnnOnGpu",type:"bool",defaultValue:!0},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2}]},{tfOpName:"Conv2DBackpropInput",category:"convolution",inputs:[{start:2,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:0,name:"outputShape",type:"number[]"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]",notSupported:!0}]},{tfOpName:"DepthwiseConv2d",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"DepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"FusedDepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]}]},{tfOpName:"Conv3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"Dilation2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"rates",name:"dilations",type:"number[]"},{tfName:"padding",name:"pad",type:"string"}]}],ko=Object.freeze(Object.defineProperty({__proto__:null,json:Io},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Ao=[{tfOpName:"Fill",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"},{start:1,name:"value",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"LinSpace",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"num",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"OneHot",category:"creation",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"depth",type:"number"},{start:2,name:"onValue",type:"number",defaultValue:1},{start:3,name:"offValue",type:"number",defaultValue:0}],attrs:[{tfName:"axis",name:"axis",type:"number",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Ones",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"OnesLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"RandomStandardNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniform",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number",defaultValue:0},{tfName:"maxval",name:"maxval",type:"number",defaultValue:1},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Range",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"step",type:"number",defaultValue:0}],attrs:[{tfName:"Tidx",name:"dtype",type:"dtype"}]},{tfOpName:"TruncatedNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"means",name:"mean",type:"number",defaultValue:0},{tfName:"stddev",name:"stdDev",type:"number",defaultValue:1},{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Zeros",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"ZerosLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Multinomial",category:"creation",inputs:[{start:0,name:"logits",type:"tensor"},{start:1,name:"numSamples",type:"number"}],attrs:[{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number"},{tfName:"T",name:"dtype",type:"dtype"},{tfName:"output_dtype",name:"output_dtype",type:"dtype"}]}],Do=Object.freeze(Object.defineProperty({__proto__:null,json:Ao},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const $o=[{tfOpName:"NonMaxSuppressionV2",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV3",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV4",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"T_threshold",name:"threshold",type:"dtype",notSupported:!0},{tfName:"pad_to_max_output_size",name:"padToMaxOutputSize",type:"bool"}]},{tfOpName:"NonMaxSuppressionV5",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"},{start:5,name:"softNmsSigma",type:"number"}]},{tfOpName:"Where",category:"dynamic",inputs:[{start:0,name:"condition",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ListDiff",category:"dynamic",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],Co=Object.freeze(Object.defineProperty({__proto__:null,json:$o},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const zo=[{tfOpName:"LowerBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"TopKV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"k",type:"number"}],attrs:[{tfName:"sorted",name:"sorted",type:"bool"}]},{tfOpName:"UpperBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"Unique",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"UniqueV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]}],Lo=Object.freeze(Object.defineProperty({__proto__:null,json:zo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Po=[{tfOpName:"PlaceholderWithDefault",category:"graph",inputs:[{start:0,name:"default",type:"tensor"}],attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Placeholder",category:"graph",attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Const",category:"graph"},{tfOpName:"Identity",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IdentityN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Snapshot",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Rank",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Size",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Shape",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"ShapeN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Print",category:"graph",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"data",type:"tensors"}],attrs:[{tfName:"message",name:"message",type:"string"},{tfName:"first_n",name:"firstN",type:"number",notSupported:!0},{tfName:"summarize",name:"summarize",type:"number",defaultValue:3}]},{tfOpName:"NoOp",category:"graph",inputs:[]},{tfOpName:"StopGradient",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"FakeQuantWithMinMaxVars",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"min",name:"min",type:"number"},{tfName:"max",name:"max",type:"number"}]}],Vo=Object.freeze(Object.defineProperty({__proto__:null,json:Po},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Fo=[{tfOpName:"HashTable",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"HashTableV2",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"LookupTableImport",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableImportV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFind",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFindV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableSize",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"LookupTableSizeV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"InitializeTable",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]},{tfOpName:"InitializeTableV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],jo=Object.freeze(Object.defineProperty({__proto__:null,json:Fo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const xo=[{tfOpName:"ResizeBilinear",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ResizeNearestNeighbor",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"CropAndResize",category:"image",inputs:[{start:0,name:"image",type:"tensor"},{start:1,name:"boxes",type:"tensor"},{start:2,name:"boxInd",type:"tensor"},{start:3,name:"cropSize",type:"number[]"}],attrs:[{tfName:"method",name:"method",type:"string"},{tfName:"extrapolation_value",name:"extrapolationValue",type:"number"}]},{tfOpName:"ImageProjectiveTransformV3",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"transforms",type:"tensor"},{start:2,name:"outputShape",type:"number[]"},{start:3,name:"fillValue",type:"number"}],attrs:[{tfName:"interpolation",name:"interpolation",type:"string"},{tfName:"fill_mode",name:"fillMode",type:"string"}]}],Ro=Object.freeze(Object.defineProperty({__proto__:null,json:xo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Bo=[{tfOpName:"Equal",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NotEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Greater",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"GreaterEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Less",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LessEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalAnd",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalNot",category:"logical",inputs:[{start:0,name:"a",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalOr",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Select",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SelectV2",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],Ho=Object.freeze(Object.defineProperty({__proto__:null,json:Bo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const qo=[{tfOpName:"_FusedMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMulV2",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Transpose",category:"matrices",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"perm",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Einsum",category:"matrices",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"equation",name:"equation",type:"string"},{tfName:"N",name:"n",type:"number",defaultValue:2},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"MatrixBandPart",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"numLower",type:"tensor"},{start:1,name:"numUpper",type:"tensor"}]}],Wo=Object.freeze(Object.defineProperty({__proto__:null,json:qo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Uo=[{tfOpName:"EuclideanNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",defaultValue:!1}]},{tfOpName:"FusedBatchNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV2",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV3",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"LRN",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"depth_radius",name:"radius",type:"number",defaultValue:5},{tfName:"bias",name:"bias",type:"number",defaultValue:1},{tfName:"alpha",name:"alpha",type:"number",defaultValue:1},{tfName:"beta",name:"beta",type:"number",defaultValue:.5}]},{tfOpName:"Softmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"LogSoftmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"SparseToDense",category:"normalization",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!0,notSupported:!0}]}],Go=Object.freeze(Object.defineProperty({__proto__:null,json:Uo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Ko=[{tfOpName:"Bincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}]},{tfOpName:"DenseBincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}],attrs:[{tfName:"binary_output",name:"binaryOutput",type:"bool"}]},{tfOpName:"Max",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Mean",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Min",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Sum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"All",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Any",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"ArgMax",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"ArgMin",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"Prod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Cumprod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]},{tfOpName:"Cumsum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]}],Jo=Object.freeze(Object.defineProperty({__proto__:null,json:Ko},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Xo=[{tfOpName:"ConcatV2",category:"slice_join",inputs:[{start:0,end:-1,name:"tensors",type:"tensors"},{start:-1,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"Concat",category:"slice_join",inputs:[{start:1,end:0,name:"tensors",type:"tensors"},{start:0,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"GatherV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"axis",type:"number",defaultValue:0}],attrs:[{tfName:"batch_dims",name:"batchDims",type:"number",defaultValue:0}]},{tfOpName:"Gather",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",notSupported:!0}]},{tfOpName:"Reverse",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"dims",type:"bool[]"}]},{tfOpName:"ReverseV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}]},{tfOpName:"Slice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"size",type:"number[]"}]},{tfOpName:"StridedSlice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"end",type:"number[]"},{start:3,name:"strides",type:"number[]"}],attrs:[{tfName:"begin_mask",name:"beginMask",type:"number",defaultValue:0},{tfName:"end_mask",name:"endMask",type:"number",defaultValue:0},{tfName:"new_axis_mask",name:"newAxisMask",type:"number",defaultValue:0},{tfName:"ellipsis_mask",name:"ellipsisMask",type:"number",defaultValue:0},{tfName:"shrink_axis_mask",name:"shrinkAxisMask",type:"number",defaultValue:0}]},{tfOpName:"Pack",category:"slice_join",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0}]},{tfOpName:"Unpack",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0},{tfName:"num",name:"num",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Tile",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"reps",type:"number[]"}]},{tfOpName:"Split",category:"slice_join",inputs:[{start:0,name:"axis",type:"number",defaultValue:0},{start:1,name:"x",type:"tensor"}],attrs:[{tfName:"num_split",name:"numOrSizeSplits",type:"number",defaultValue:1}]},{tfOpName:"SplitV",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"numOrSizeSplits",type:"number[]"},{start:2,name:"axis",type:"number",defaultValue:0}]},{tfOpName:"ScatterNd",category:"slice_join",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"shape",type:"number[]"}]},{tfOpName:"GatherNd",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}]},{tfOpName:"SparseToDense",category:"slice_join",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!1,notSupported:!0}]},{tfOpName:"TensorScatterUpdate",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],Yo=Object.freeze(Object.defineProperty({__proto__:null,json:Xo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Zo=[{tfOpName:"SparseFillEmptyRows",category:"sparse",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"denseShape",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}]},{tfOpName:"SparseReshape",category:"sparse",inputs:[{start:0,name:"inputIndices",type:"tensor"},{start:1,name:"inputShape",type:"tensor"},{start:2,name:"newShape",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SparseSegmentMean",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]},{tfOpName:"SparseSegmentSum",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]}],Qo=Object.freeze(Object.defineProperty({__proto__:null,json:Zo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const Mo=[{tfOpName:"FFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"RFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]},{tfOpName:"IRFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]}],eu=Object.freeze(Object.defineProperty({__proto__:null,json:Mo},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const tu=[{tfOpName:"StaticRegexReplace",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"pattern",name:"pattern",type:"string"},{tfName:"rewrite",name:"rewrite",type:"string"},{tfName:"replace_global",name:"replaceGlobal",type:"bool"}]},{tfOpName:"StringNGrams",category:"string",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"dataSplits",type:"tensor"}],attrs:[{tfName:"separator",name:"separator",type:"string"},{tfName:"ngram_widths",name:"nGramWidths",type:"number[]"},{tfName:"left_pad",name:"leftPad",type:"string"},{tfName:"right_pad",name:"rightPad",type:"string"},{tfName:"pad_width",name:"padWidth",type:"number"},{tfName:"preserve_short_sequences",name:"preserveShortSequences",type:"bool"}],outputs:["ngrams","ngrams_splits"]},{tfOpName:"StringSplit",category:"string",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"delimiter",type:"tensor"}],attrs:[{tfName:"skip_empty",name:"skipEmpty",type:"bool"}],outputs:["indices","values","shape"]},{tfOpName:"StringToHashBucketFast",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"num_buckets",name:"numBuckets",type:"number"}]}],au=Object.freeze(Object.defineProperty({__proto__:null,json:tu},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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
 */const su=[{tfOpName:"Cast",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"SrcT",name:"sdtype",type:"dtype",notSupported:!0},{tfName:"DstT",name:"dtype",type:"dtype"}]},{tfOpName:"ExpandDims",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"MirrorPad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"mode",name:"mode",type:"string"}]},{tfOpName:"Pad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"constant_value",name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"PadV2",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"},{start:2,name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"Reshape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"Squeeze",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"axis",tfDeprecatedName:"squeeze_dims",name:"axis",type:"number[]"}]},{tfOpName:"SpaceToBatchND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"paddings",type:"number[]"}]},{tfOpName:"BatchToSpaceND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"crops",type:"number[]"}]},{tfOpName:"DepthToSpace",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"block_size",name:"blockSize",type:"number"},{tfName:"data_format",name:"dataFormat",type:"string"}]},{tfOpName:"BroadcastTo",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}],attrs:[]},{tfOpName:"BroadcastArgs",category:"transformation",inputs:[{start:0,name:"s0",type:"tensor"},{start:1,name:"s1",type:"tensor"}],attrs:[]}],ru=Object.freeze(Object.defineProperty({__proto__:null,json:su},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */class ht{static get Instance(){return this._instance||(this._instance=new this)}constructor(){const e=[So,vo,Eo,ko,Do,Co,Lo,Vo,jo,Ro,Ho,Wo,Go,Jo,Yo,Qo,eu,au,ru],t=[].concat(...e.map(s=>s.json));this.opMappers=t.reduce((s,r)=>(s[r.tfOpName]=r,s),{})}transformGraph(e,t={}){const s=e.node,r=[],i=[],o=[],u=s.reduce((y,h)=>(y[h.name]=this.mapNode(h),h.op.startsWith("Placeholder")?r.push(y[h.name]):h.op==="Const"?i.push(y[h.name]):(h.input==null||h.input.length===0)&&o.push(y[h.name]),y),{});let p=[];const m=[];let l={},c={};t!=null&&(l=this.mapSignatureEntries(t.inputs),c=this.mapSignatureEntries(t.outputs));const d=Object.keys(u);d.forEach(y=>{const h=u[y];h.inputNames.forEach((N,_)=>{const[A,,T]=W(N),I=u[A];if(I.outputs!=null){const $=I.outputs.indexOf(T);if($!==-1){const w=`${A}:${$}`;h.inputNames[_]=w}}h.inputs.push(I),I.children.push(h)})}),Object.keys(c).length===0?d.forEach(y=>{const h=u[y];h.children.length===0&&m.push(h)}):Object.keys(c).forEach(y=>{const[h]=W(y),N=u[h];N!=null&&(N.signatureKey=c[y],m.push(N))}),Object.keys(l).length>0?Object.keys(l).forEach(y=>{const[h]=W(y),N=u[h];N&&(N.signatureKey=l[y],p.push(N))}):p=r;let f={};e.library!=null&&e.library.function!=null&&(f=e.library.function.reduce((y,h)=>(y[h.signature.name]=this.mapFunction(h),y),{}));const g={nodes:u,inputs:p,outputs:m,weights:i,placeholders:r,signature:t,functions:f};return o.length>0&&(g.initNodes=o),g}mapSignatureEntries(e){return Object.keys(e||{}).reduce((t,s)=>(t[e[s].name]=s,t),{})}mapNode(e){const t=Ut(e.op)||this.opMappers[e.op]||{};e.attr==null&&(e.attr={});const s={name:e.name,op:e.op,category:t.category,inputNames:(e.input||[]).map(r=>r.startsWith("^")?r.slice(1):r),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:e.attr,outputs:t.outputs};return t.inputs!=null&&(s.inputParams=t.inputs.reduce((r,i)=>(r[i.name]={type:i.type,inputIndexStart:i.start,inputIndexEnd:i.end},r),{})),t.attrs!=null&&(s.attrParams=t.attrs.reduce((r,i)=>{const o=i.type;let u;switch(i.type){case"string":u=Ve(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=Ve(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"string[]":u=qe(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=qe(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"number":u=je(e.attr,i.tfName,i.defaultValue||0),u===void 0&&i.tfDeprecatedName&&(u=je(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"number[]":u=He(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=He(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"bool":u=Fe(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=Fe(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"bool[]":u=Ue(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=Ue(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"shape":u=Be(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=Be(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"shape[]":u=We(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=We(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"dtype":u=xe(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=xe(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"dtype[]":u=Re(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=Re(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"func":u=ft(e.attr,i.tfName,i.defaultValue),u===void 0&&i.tfDeprecatedName&&(u=ft(e.attr,i.tfDeprecatedName,i.defaultValue));break;case"tensor":case"tensors":break;default:throw new Error(`Unsupported param type: ${i.type} for op: ${e.op}`)}return r[i.name]={value:u,type:o},r},{})),s}mapFunction(e){const t=e.nodeDef,s=[],r=[];let i={};t!=null&&(i=t.reduce((c,d)=>(c[d.name]=this.mapNode(d),d.op==="Const"&&r.push(c[d.name]),c),{}));const o=[],u=[];e.signature.inputArg.forEach(c=>{const[d]=W(c.name),f={name:d,op:"Placeholder",inputs:[],inputNames:[],category:"graph",inputParams:{},attrParams:{dtype:{value:et(c.type),type:"dtype"}},children:[]};f.signatureKey=c.name,o.push(f),i[d]=f}),Object.keys(i).forEach(c=>{const d=i[c];d.inputNames.forEach((f,g)=>{const[y,,h]=W(f),N=i[y];if(N.outputs!=null){const _=N.outputs.indexOf(h);if(_!==-1){const A=`${y}:${_}`;d.inputNames[g]=A}}d.inputs.push(N),N.children.push(d)})});const m=e.ret;e.signature.outputArg.forEach(c=>{const[d,f]=W(m[c.name]),g=i[d];g!=null&&(g.defaultOutput=f,u.push(g))});const l=this.mapArgsToSignature(e);return{nodes:i,inputs:o,outputs:u,weights:r,placeholders:s,signature:l}}mapArgsToSignature(e){return{methodName:e.signature.name,inputs:e.signature.inputArg.reduce((t,s)=>(t[s.name]=this.mapArgToTensorInfo(s),t),{}),outputs:e.signature.outputArg.reduce((t,s)=>(t[s.name]=this.mapArgToTensorInfo(s,e.ret),t),{})}}mapArgToTensorInfo(e,t){let s=e.name;return t!=null&&(s=t[s]),{name:s,dtype:e.type}}}function nu(a){const e=Q().global;if(typeof e.atob<"u")return e.atob(a);if(typeof Buffer<"u")return new Buffer(a,"base64").toString();throw new Error("Unable to decode base64 in this environment. Missing built-in atob() or Buffer()")}function Gt(a,e){const t=Array.isArray(a)?String.fromCharCode.apply(null,a):nu(a);return e?t:t.toLowerCase()}function Ve(a,e,t,s=!1){const r=a[e];return r!=null?Gt(r.s,s):t}function Fe(a,e,t){const s=a[e];return s?s.b:t}function je(a,e,t){const s=a[e]||{},r=s.i!=null?s.i:s.f!=null?s.f:t;return typeof r=="number"?r:parseInt(r,10)}function et(a){switch(typeof a=="string"&&(a=q[a]),a){case q.DT_FLOAT:case q.DT_HALF:return"float32";case q.DT_INT32:case q.DT_INT64:case q.DT_INT8:case q.DT_UINT8:return"int32";case q.DT_BOOL:return"bool";case q.DT_DOUBLE:return"float32";case q.DT_STRING:return"string";default:return null}}function ft(a,e,t){const s=a[e];return s&&s.func?s.func.name:t}function xe(a,e,t){const s=a[e];return s&&s.type?et(s.type):t}function Re(a,e,t){const s=a[e];return s&&s.list&&s.list.type?s.list.type.map(r=>et(r)):t}function Kt(a){if(!a.unknownRank)return a.dim!=null?a.dim.map(e=>typeof e.size=="number"?e.size:parseInt(e.size,10)):[]}function Be(a,e,t){const s=a[e];return s&&s.shape?Kt(s.shape):t}function He(a,e,t){const s=a[e];return s?((s.list.f&&s.list.f.length?s.list.f:s.list.i)||[]).map(r=>typeof r=="number"?r:parseInt(r,10)):t}function qe(a,e,t,s=!1){const r=a[e];return r&&r.list&&r.list.s?r.list.s.map(i=>Gt(i,s)):t}function We(a,e,t){const s=a[e];return s&&s.list&&s.list.shape?s.list.shape.map(r=>Kt(r)):t}function Ue(a,e,t){const s=a[e];return s&&s.list&&s.list.b?s.list.b:t}/**
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
 */class iu{constructor(e,t,s){this.node=e,this.tensorMap=t,this.context=s,this.inputs=[],this.attrs={},this.inputs=e.inputNames.map(r=>this.getInput(r)),e.rawAttrs!=null&&(this.attrs=Object.keys(e.rawAttrs).reduce((r,i)=>(r[i]=this.getAttr(i),r),{}))}getInput(e){return L(e,this.tensorMap,this.context)}getAttr(e,t){const s=this.node.rawAttrs[e];if(s.tensor!=null)return L(e,this.tensorMap,this.context);if(s.i!=null||s.f!=null)return je(this.node.rawAttrs,e,t);if(s.s!=null)return Ve(this.node.rawAttrs,e,t);if(s.b!=null)return Fe(this.node.rawAttrs,e,t);if(s.shape!=null)return Be(this.node.rawAttrs,e,t);if(s.type!=null)return xe(this.node.rawAttrs,e,t);if(s.list!=null){if(s.list.i!=null||s.list.f!=null)return He(this.node.rawAttrs,e,t);if(s.list.s!=null)return qe(this.node.rawAttrs,e,t);if(s.list.shape!=null)return We(this.node.rawAttrs,e,t);if(s.list.b!=null)return Ue(this.node.rawAttrs,e,t);if(s.list.type!=null)return Re(this.node.rawAttrs,e,t)}return t}}/**
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
 */const P=Object.freeze(Object.defineProperty({__proto__:null,OP_SCOPE_SUFFIX:Ha,abs:qa,acos:ys,acosh:gs,add:ae,addN:Pn,all:Ns,any:bs,argMax:Ts,argMin:ws,asin:Ss,asinh:_s,atan:vs,atan2:Os,atanh:Es,avgPool:Is,avgPool3d:ks,basicLSTMCell:Fn,batchNorm:As,batchNorm2d:Ds,batchNorm3d:$s,batchNorm4d:Cs,batchToSpaceND:zs,bincount:Ls,booleanMaskAsync:Ri,broadcastArgs:xn,broadcastTo:Wa,buffer:wt,cast:Ua,ceil:Ps,clipByValue:Vs,clone:At,complex:Ga,concat:Ie,concat1d:Fs,concat2d:js,concat3d:xs,concat4d:Rs,conv1d:Bs,conv2d:Hs,conv2dTranspose:qs,conv3d:Ws,conv3dTranspose:Us,cos:Gs,cosh:Ks,cosineWindow:Js,cumprod:Xs,cumsum:Ys,denseBincount:Zs,depthToSpace:Qs,depthwiseConv2d:zt,diag:Bn,dilation2d:Ms,div:vt,divNoNan:er,dot:tr,dropout:ar,einsum:qn,elu:Ka,enclosingPowerOfTwo:sr,equal:rr,erf:nr,euclideanNorm:ir,exp:or,expandDims:ur,expm1:pr,eye:mr,fft:lr,fill:Ja,floor:cr,floorDiv:Xa,fused:eo,gather:Ct,gatherND:Xi,greater:dr,greaterEqual:hr,ifft:fr,imag:yr,image:Le,inTopKAsync:Zi,irfft:gr,isFinite:Nr,isInf:br,isNaN:Tr,leakyRelu:Ya,less:wr,lessEqual:Sr,linalg:_r,linspace:Wn,localResponseNormalization:vr,log:Or,log1p:Er,logSigmoid:Ir,logSoftmax:kr,logSumExp:Ar,logicalAnd:Dr,logicalNot:$r,logicalOr:Cr,logicalXor:zr,losses:Lr,lowerBound:Gn,matMul:Y,max:Pr,maxPool:Vr,maxPool3d:Fr,maxPoolWithArgmax:Jn,maximum:Za,mean:jr,meshgrid:Xn,min:xr,minimum:Rr,mirrorPad:Br,mod:Hr,moments:qr,movingAverage:Hi,mul:ue,multiRNNCell:Zn,multinomial:Mn,neg:Wr,norm:Ur,notEqual:Gr,oneHot:Lt,ones:oe,onesLike:Kr,op:v,outerProduct:ti,pad:me,pad1d:si,pad2d:ni,pad3d:oi,pad4d:pi,pool:Jr,pow:Ot,prelu:Qa,print:Ma,prod:Xr,raggedGather:li,raggedRange:di,raggedTensorToTensor:fi,rand:gi,randomGamma:bi,randomNormal:Dt,randomStandardNormal:wi,randomUniform:Yr,range:Zr,real:Qr,reciprocal:Mr,relu:es,relu6:ts,reshape:k,reverse:le,reverse1d:_i,reverse2d:Oi,reverse3d:Ii,reverse4d:Ai,rfft:en,round:tn,rsqrt:an,scalar:K,scatterND:Wi,searchSorted:Ze,selu:sn,separableConv2d:rn,setdiff1dAsync:$i,sigmoid:fe,sign:nn,signal:on,sin:un,sinh:pn,slice:Z,slice1d:mn,slice2d:ln,slice3d:cn,slice4d:dn,softmax:hn,softplus:fn,spaceToBatchND:yn,sparse:gn,sparseToDense:Ki,spectral:Nn,split:bn,sqrt:as,square:ss,squaredDifference:Tn,squeeze:$t,stack:re,step:rs,stridedSlice:wn,string:Sn,sub:ye,sum:ns,tan:_n,tanh:ze,tensor:se,tensor1d:Pt,tensor2d:vn,tensor3d:Ft,tensor4d:Ci,tensor5d:zi,tensor6d:Li,tensorScatterUpdate:Vi,tile:On,topk:En,transpose:In,truncatedNormal:kn,unique:An,unsortedSegmentSum:Dn,unstack:ce,upperBound:Fi,variable:$n,where:Cn,whereAsync:jt,zeros:Vt,zerosLike:is},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const ou=(a,e,t,s=P)=>{switch(a.op){case"BiasAdd":case"AddV2":case"Add":return[s.add(n("a",a,e,t),n("b",a,e,t))];case"AddN":return[s.addN(n("tensors",a,e,t))];case"FloorMod":case"Mod":return[s.mod(n("a",a,e,t),n("b",a,e,t))];case"Mul":return[s.mul(n("a",a,e,t),n("b",a,e,t))];case"RealDiv":case"Div":return[s.div(n("a",a,e,t),n("b",a,e,t))];case"DivNoNan":return[s.divNoNan(n("a",a,e,t),n("b",a,e,t))];case"FloorDiv":return[s.floorDiv(n("a",a,e,t),n("b",a,e,t))];case"Sub":return[s.sub(n("a",a,e,t),n("b",a,e,t))];case"Minimum":return[s.minimum(n("a",a,e,t),n("b",a,e,t))];case"Maximum":return[s.maximum(n("a",a,e,t),n("b",a,e,t))];case"Pow":return[s.pow(n("a",a,e,t),n("b",a,e,t))];case"SquaredDifference":return[s.squaredDifference(n("a",a,e,t),n("b",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const uu=(a,e,t,s=P)=>{switch(a.op){case"Abs":case"ComplexAbs":return[s.abs(n("x",a,e,t))];case"Acos":return[s.acos(n("x",a,e,t))];case"Acosh":return[s.acosh(n("x",a,e,t))];case"Asin":return[s.asin(n("x",a,e,t))];case"Asinh":return[s.asinh(n("x",a,e,t))];case"Atan":return[s.atan(n("x",a,e,t))];case"Atan2":return[s.atan2(n("x",a,e,t),n("y",a,e,t))];case"Atanh":return[s.atanh(n("x",a,e,t))];case"Ceil":return[s.ceil(n("x",a,e,t))];case"Complex":return[s.complex(n("real",a,e,t),n("imag",a,e,t))];case"Cos":return[s.cos(n("x",a,e,t))];case"Cosh":return[s.cosh(n("x",a,e,t))];case"Elu":return[s.elu(n("x",a,e,t))];case"Erf":return[s.erf(n("x",a,e,t))];case"Exp":return[s.exp(n("x",a,e,t))];case"Expm1":return[s.expm1(n("x",a,e,t))];case"Floor":return[s.floor(n("x",a,e,t))];case"Log":return[s.log(n("x",a,e,t))];case"Log1p":return[s.log1p(n("x",a,e,t))];case"Imag":return[s.imag(n("x",a,e,t))];case"Neg":return[s.neg(n("x",a,e,t))];case"Reciprocal":return[s.reciprocal(n("x",a,e,t))];case"Real":return[s.real(n("x",a,e,t))];case"Relu":return[s.relu(n("x",a,e,t))];case"Round":return[s.round(n("x",a,e,t))];case"Selu":return[s.selu(n("x",a,e,t))];case"Sigmoid":return[s.sigmoid(n("x",a,e,t))];case"Sin":return[s.sin(n("x",a,e,t))];case"Sign":return[s.sign(n("x",a,e,t))];case"Sinh":return[s.sinh(n("x",a,e,t))];case"Softplus":return[s.softplus(n("x",a,e,t))];case"Sqrt":return[s.sqrt(n("x",a,e,t))];case"Square":return[s.square(n("x",a,e,t))];case"Tanh":return[s.tanh(n("x",a,e,t))];case"Tan":return[s.tan(n("x",a,e,t))];case"ClipByValue":return[s.clipByValue(n("x",a,e,t),n("clipValueMin",a,e,t),n("clipValueMax",a,e,t))];case"Relu6":return[s.relu6(n("x",a,e,t))];case"Rsqrt":return[s.rsqrt(L(a.inputNames[0],e,t))];case"Prod":return[s.prod(n("x",a,e,t),n("axes",a,e,t))];case"LeakyRelu":return[s.leakyRelu(n("x",a,e,t),n("alpha",a,e,t))];case"Prelu":return[s.prelu(n("x",a,e,t),n("alpha",a,e,t))];case"IsNan":return[s.isNaN(L(a.inputNames[0],e,t))];case"IsInf":return[s.isInf(L(a.inputNames[0],e,t))];case"IsFinite":return[s.isFinite(L(a.inputNames[0],e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */function B(a,e,t=""){if(!(typeof a=="number"||typeof e=="number")){S(a.length===e.length,()=>t+` Shapes ${a} and ${e} must match`);for(let s=0;s<a.length;s++){const r=a[s],i=e[s];S(r<0||i<0||r===i,()=>t+` Shapes ${a} and ${e} must match`)}}}function yt(a){return!(typeof a=="number"||a.some(e=>e<0))}function ie(a,e,t){let s=Ge(a,t);const r=!yt(s);if(r&&e.length===0)throw new Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${s}`);if(r&&e.forEach(i=>{s=Ge(i.shape,s)}),!yt(s))throw new Error(`Non-fully-defined elementShape: ${s}`);return s}function Ge(a,e){if(typeof a=="number")return e;if(typeof e=="number")return a;if(a.length!==e.length)throw new Error(`Incompatible ranks during merge: ${a} vs. ${e}`);const t=[];for(let s=0;s<a.length;++s){const r=a[s],i=e[s];if(r>=0&&i>=0&&r!==i)throw new Error(`Incompatible shape during merge: ${a} vs. ${e}`);t[s]=r>=0?r:i}return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */class pu{constructor(e,t,s,r,i,o,u){this.name=e,this.dtype=t,this.maxSize=s,this.elementShape=r,this.identicalElementShapes=i,this.dynamicSize=o,this.clearAfterRead=u,this.tensors=[],this.closed_=!1,this.idTensor=K(0),G(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(e){this.tensors.forEach(t=>{(e==null||!e.has(t.tensor.id))&&t.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(e){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||e>=this.size())throw new Error(`Tried to read from index ${e}, but array size is: ${this.size()}`);const t=this.tensors[e];if(t.cleared)throw new Error(`TensorArray ${this.name}: Could not read index ${e} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(t.cleared=!0),t.read=!0,t.tensor}readMany(e){return e.map(t=>this.read(t))}write(e,t){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||!this.dynamicSize&&e>=this.maxSize)throw new Error(`Tried to write to index ${e}, but array is not resizeable and size is: ${this.maxSize}`);const s=this.tensors[e]||{};if(t.dtype!==this.dtype)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e},
          because the value dtype is ${t.dtype}, but TensorArray dtype is ${this.dtype}.`);if(this.size()===0&&(this.elementShape==null||this.elementShape.length===0)&&(this.elementShape=t.shape),B(this.elementShape,t.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${e}.`),s.read)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been read.`);if(s.written)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been written.`);s.tensor=t,G(t),s.written=!0,this.tensors[e]=s}writeMany(e,t){if(e.length!==t.length)throw new Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${e.length} is not the same as tensors size: ${t.length}.`);e.forEach((s,r)=>this.write(s,t[r]))}gather(e,t){if(t&&t!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${t}`);if(e)e=e.slice(0,this.size());else{e=[];for(let r=0;r<this.size();r++)e.push(r)}if(e.length===0)return se([],[0].concat(this.elementShape));const s=this.readMany(e);return B(this.elementShape,s[0].shape,"TensorArray shape mismatch: "),re(s,0)}concat(e){if(e&&e!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${e}`);if(this.size()===0)return se([],[0].concat(this.elementShape));const t=[];for(let r=0;r<this.size();r++)t.push(r);const s=this.readMany(t);return B(this.elementShape,s[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${s[0].shape})`),Ie(s,0)}scatter(e,t){if(t.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);if(e.length!==t.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${t.shape[0]}`);const s=Math.max(...e);if(!this.dynamicSize&&s>=this.maxSize)throw new Error(`Max index must be < array size (${s}  vs. ${this.maxSize})`);this.writeMany(e,ce(t,0))}split(e,t){if(t.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);let s=0;const r=e.map(p=>(s+=p,s));if(s!==t.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${s}, and tensor's shape is: ${t.shape}`);if(!this.dynamicSize&&e.length!==this.maxSize)throw new Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${e.length}), and the TensorArray is not marked as dynamically resizeable`);const i=s===0?0:t.size/s,o=[];R(()=>{t=k(t,[1,s,i]);for(let p=0;p<e.length;++p){const l=[0,p===0?0:r[p-1],0],c=[1,e[p],i];o[p]=k(Z(t,l,c),this.elementShape)}return o});const u=[];for(let p=0;p<e.length;p++)u[p]=p;this.writeMany(u,o)}}/**
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
 */class ee{get id(){return this.idTensor.id}constructor(e,t,s,r=-1){this.tensors=e,this.elementShape=t,this.elementDtype=s,e!=null&&e.forEach(i=>{if(s!==i.dtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${i.dtype}`);B(t,i.shape,"TensorList shape mismatch: "),G(i)}),this.idTensor=K(0),this.maxNumElements=r,G(this.idTensor)}copy(){return new ee([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(e){this.tensors.forEach(t=>{(e==null||!e.has(t.id))&&t.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(e,t,s=-1){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(s!==-1&&this.tensors.length!==s)throw new Error(`Operation expected a list with ${s} elements but got a list with ${this.tensors.length} elements.`);B(e,this.elementShape,"TensorList shape mismatch: ");const r=ie(this.elementShape,this.tensors,e);return R(()=>{const i=this.tensors.map(o=>k(o,r));return re(i,0)})}popBack(e,t){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(this.size()===0)throw new Error("Trying to pop from an empty list.");const s=ie(this.elementShape,this.tensors,e),r=this.tensors.pop();return r.kept=!1,B(r.shape,e,"TensorList shape mismatch: "),k(r,s)}pushBack(e){if(e.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${e.dtype}, but list elements ${this.elementDtype}`);if(B(e.shape,this.elementShape,"TensorList shape mismatch: "),this.maxNumElements===this.size())throw new Error("Trying to push element into a full list.");G(e),this.tensors.push(e)}resize(e){if(e<0)throw new Error(`TensorListResize expects size to be non-negative. Got: ${e}`);if(this.maxNumElements!==-1&&e>this.maxNumElements)throw new Error(`TensorListResize input size ${e} is greater maxNumElement ${this.maxNumElements}.`);const t=new ee([],this.elementShape,this.elementDtype,this.maxNumElements);t.tensors.length=e;for(let s=0;s<Math.min(this.tensors.length,e);++s)t.tensors[s]=this.tensors[s];return t}getItem(e,t,s){if(s!==this.elementDtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${this.elementDtype}`);if(e<0||e>this.tensors.length)throw new Error(`Trying to access element ${e} in a list with ${this.tensors.length} elements.`);if(this.tensors[e]==null)throw new Error(`element at index ${e} is null.`);B(this.tensors[e].shape,t,"TensorList shape mismatch: ");const r=ie(this.elementShape,this.tensors,t);return k(this.tensors[e],r)}setItem(e,t){if(t.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t.dtype}, but list elements ${this.elementDtype}`);if(e<0||this.maxNumElements!==-1&&e>=this.maxNumElements)throw new Error(`Trying to set element ${e} in a list with max ${this.maxNumElements} elements.`);B(this.elementShape,t.shape,"TensorList shape mismatch: "),G(t),this.tensors[e]!=null&&(this.tensors[e].kept=!1),this.tensors[e]=t}gather(e,t,s){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);B(this.elementShape,s,"TensorList shape mismatch: "),e=e.slice(0,this.size());const r=ie(this.elementShape,this.tensors,s);return e.length===0?se([],[0].concat(r)):R(()=>{const i=e.map(o=>k(this.tensors[o],r));return re(i,0)})}concat(e,t){if(e&&e!==this.elementDtype)throw new Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${e}`);B(this.elementShape,t,"TensorList shape mismatch: ");const s=ie(this.elementShape,this.tensors,t);return this.size()===0?se([],[0].concat(s)):R(()=>{const r=this.tensors.map(i=>k(i,s));return Ie(r,0)})}}function mu(a,e,t){const s=a.dtype;if(a.shape.length<1)throw new Error(`Tensor must be at least a vector, but saw shape: ${a.shape}`);if(a.dtype!==t)throw new Error(`Invalid data types; op elements ${a.dtype}, but list elements ${t}`);const r=a.shape.slice(1);B(r,e,"TensorList shape mismatch: ");const i=ce(a);return new ee(i,e,s)}function lu(a,e,t,s){return new ee([],a,e,s)}function cu(a,e,t,s){if(e.length!==a.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${a.shape[0]}`);const r=Math.max(...e);if(s!=null&&s!==-1&&r>=s)throw new Error(`Max index must be < array size (${r}  vs. ${s})`);const i=new ee([],t,a.dtype,s),o=ce(a,0);return e.forEach((u,p)=>{i.setItem(u,o[p])}),i}function du(a,e,t){let s=0;const r=e.map(l=>(s+=l,s));if(s!==a.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${s}, and tensor's shape is: ${a.shape}`);const i=a.shape.slice(1),o=Ge(i,t),u=s===0?0:a.size/s,p=R(()=>{const l=[];a=k(a,[1,s,u]);for(let c=0;c<e.length;++c){const f=[0,c===0?0:r[c-1],0],g=[1,e[c],u];l[c]=k(Z(a,f,g),o)}return a.dispose(),l}),m=new ee([],t,a.dtype,e.length);for(let l=0;l<p.length;l++)m.setItem(l,p[l]);return m}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const hu=async(a,e,t)=>{switch(a.op){case"If":case"StatelessIf":{const s=n("thenBranch",a,e,t),r=n("elseBranch",a,e,t),i=n("cond",a,e,t),o=n("args",a,e,t);return(await i.data())[0]?t.functionMap[s].executeFunctionAsync(o,t.tensorArrayMap,t.tensorListMap):t.functionMap[r].executeFunctionAsync(o,t.tensorArrayMap,t.tensorListMap)}case"While":case"StatelessWhile":{const s=n("body",a,e,t),r=n("cond",a,e,t),i=n("args",a,e,t),o=await t.functionMap[r].executeFunctionAsync(i,t.tensorArrayMap,t.tensorListMap),u=i.map(l=>l.id);let p=await o[0].data();o.forEach(l=>{!l.kept&&u.indexOf(l.id)===-1&&l.dispose()});let m=i;for(;p[0];){const l=m;m=await t.functionMap[s].executeFunctionAsync(m,t.tensorArrayMap,t.tensorListMap);const c=m.map(f=>f.id);l.forEach(f=>{!f.kept&&u.indexOf(f.id)===-1&&c.indexOf(f.id)===-1&&f.dispose()});const d=await t.functionMap[r].executeFunctionAsync(m,t.tensorArrayMap,t.tensorListMap);p=await d[0].data(),d.forEach(f=>{!f.kept&&u.indexOf(f.id)===-1&&c.indexOf(f.id)===-1&&f.dispose()})}return m}case"LoopCond":{const s=n("pred",a,e,t);return[U(s)]}case"Switch":{const s=n("pred",a,e,t);let r=n("data",a,e,t);return r.kept||(r=U(r)),(await s.data())[0]?[void 0,r]:[r,void 0]}case"Merge":{const s=a.inputNames.find(r=>L(r,e,t)!==void 0);if(s){const r=L(s,e,t);return[U(r)]}return}case"Enter":{const s=n("frameName",a,e,t),r=n("tensor",a,e,t);return t.enterFrame(s),[U(r)]}case"Exit":{const s=n("tensor",a,e,t);return t.exitFrame(),[U(s)]}case"NextIteration":{const s=n("tensor",a,e,t);return t.nextIteration(),[U(s)]}case"TensorArrayV3":{const s=n("size",a,e,t),r=n("dtype",a,e,t),i=n("elementShape",a,e,t),o=n("dynamicSize",a,e,t),u=n("clearAfterRead",a,e,t),p=n("identicalElementShapes",a,e,t),m=n("name",a,e,t),l=new pu(m,r,s,i,p,o,u);return t.addTensorArray(l),[l.idTensor,K(1)]}case"TensorArrayWriteV3":{const s=n("tensorArrayId",a,e,t),r=n("index",a,e,t),i=n("tensor",a,e,t),o=t.getTensorArray(s.id);return o.write(r,i),[o.idTensor]}case"TensorArrayReadV3":{const s=n("tensorArrayId",a,e,t),r=n("index",a,e,t);return[t.getTensorArray(s.id).read(r)]}case"TensorArrayGatherV3":{const s=n("tensorArrayId",a,e,t),r=n("indices",a,e,t),i=n("dtype",a,e,t);return[t.getTensorArray(s.id).gather(r,i)]}case"TensorArrayScatterV3":{const s=n("tensorArrayId",a,e,t),r=n("indices",a,e,t),i=n("tensor",a,e,t),o=t.getTensorArray(s.id);return o.scatter(r,i),[o.idTensor]}case"TensorArrayConcatV3":{const s=n("tensorArrayId",a,e,t),r=t.getTensorArray(s.id),i=n("dtype",a,e,t);return[r.concat(i)]}case"TensorArraySplitV3":{const s=n("tensorArrayId",a,e,t),r=n("tensor",a,e,t),i=n("lengths",a,e,t),o=t.getTensorArray(s.id);return o.split(i,r),[o.idTensor]}case"TensorArraySizeV3":{const s=n("tensorArrayId",a,e,t),r=t.getTensorArray(s.id);return[K(r.size(),"int32")]}case"TensorArrayCloseV3":{const s=n("tensorArrayId",a,e,t),r=t.getTensorArray(s.id);return r.clearAndClose(),[r.idTensor]}case"TensorListSetItem":{const s=n("tensorListId",a,e,t),r=n("index",a,e,t),i=n("tensor",a,e,t),o=t.getTensorList(s.id);return o.setItem(r,i),[o.idTensor]}case"TensorListGetItem":{const s=n("tensorListId",a,e,t),r=n("index",a,e,t),i=n("elementShape",a,e,t),o=n("elementDType",a,e,t);return[t.getTensorList(s.id).getItem(r,i,o)]}case"TensorListScatterV2":case"TensorListScatter":{const s=n("indices",a,e,t),r=n("tensor",a,e,t),i=n("elementShape",a,e,t),o=n("numElements",a,e,t),u=cu(r,s,i,o);return t.addTensorList(u),[u.idTensor]}case"TensorListReserve":case"EmptyTensorList":{const s=n("elementShape",a,e,t),r=n("elementDType",a,e,t);let i;a.op==="TensorListReserve"?i="numElements":i="maxNumElements";const o=n(i,a,e,t),u=a.op==="TensorListReserve"?-1:o,p=lu(s,r,o,u);return t.addTensorList(p),[p.idTensor]}case"TensorListGather":{const s=n("tensorListId",a,e,t),r=n("indices",a,e,t),i=n("elementShape",a,e,t),o=n("elementDType",a,e,t);return[t.getTensorList(s.id).gather(r,o,i)]}case"TensorListStack":{const s=n("tensorListId",a,e,t),r=n("elementShape",a,e,t),i=n("elementDType",a,e,t),o=n("numElements",a,e,t);return[t.getTensorList(s.id).stack(r,i,o)]}case"TensorListFromTensor":{const s=n("tensor",a,e,t),r=n("elementShape",a,e,t),i=n("elementDType",a,e,t),o=mu(s,r,i);return t.addTensorList(o),[o.idTensor]}case"TensorListConcat":case"TensorListConcatV2":{const s=n("tensorListId",a,e,t),r=t.getTensorList(s.id),i=n("dtype",a,e,t),o=n("elementShape",a,e,t);return[r.concat(i,o)]}case"TensorListPushBack":{const s=n("tensorListId",a,e,t),r=n("tensor",a,e,t),i=t.getTensorList(s.id);return i.pushBack(r),[i.idTensor]}case"TensorListPopBack":{const s=n("tensorListId",a,e,t),r=n("elementShape",a,e,t),i=n("elementDType",a,e,t);return[t.getTensorList(s.id).popBack(r,i)]}case"TensorListSplit":{const s=n("tensor",a,e,t),r=n("elementShape",a,e,t),i=n("lengths",a,e,t),o=du(s,i,r);return t.addTensorList(o),[o.idTensor]}case"TensorListLength":{const s=n("tensorListId",a,e,t),r=t.getTensorList(s.id);return[K(r.size(),"int32")]}case"TensorListResize":{const s=n("tensorListId",a,e,t),r=n("size",a,e,t),o=t.getTensorList(s.id).resize(r);return t.addTensorList(o),[o.idTensor]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function gt(a,e,t){const[s,r]=n("fusedOps",a,e,t),i=s==="biasadd",o=!i,u=r==="prelu",p=s==="fusedbatchnorm",m=n("numArgs",a,e,t);if(i){if(u&&m!==2)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!u&&i&&m!==1)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.")}if(p)throw new Error("FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported");const l=n("strides",a,e,t),c=ge(a,e,t),d=n("dataFormat",a,e,t).toUpperCase(),f=n("dilations",a,e,t);let[g,y]=n("args",a,e,t);o&&(y=g,g=void 0);const h=n("leakyreluAlpha",a,e,t);return{stride:l,pad:c,dataFormat:d,dilations:f,biasArg:g,preluArg:y,activationFunc:r,leakyreluAlpha:h}}const fu=(a,e,t,s=P)=>{switch(a.op){case"Conv1D":{const r=n("stride",a,e,t),i=n("pad",a,e,t),o=n("dataFormat",a,e,t).toUpperCase(),u=n("dilation",a,e,t);return[s.conv1d(n("x",a,e,t),n("filter",a,e,t),r,i,o,u)]}case"Conv2D":{const r=n("strides",a,e,t),i=ge(a,e,t),o=n("dataFormat",a,e,t).toUpperCase(),u=n("dilations",a,e,t);return[s.conv2d(n("x",a,e,t),n("filter",a,e,t),[r[1],r[2]],i,o,[u[1],u[2]])]}case"_FusedConv2D":{const{stride:r,pad:i,dataFormat:o,dilations:u,biasArg:p,preluArg:m,activationFunc:l,leakyreluAlpha:c}=gt(a,e,t);return[s.fused.conv2d({x:n("x",a,e,t),filter:n("filter",a,e,t),strides:[r[1],r[2]],pad:i,dataFormat:o,dilations:[u[1],u[2]],bias:p,activation:l,preluActivationWeights:m,leakyreluAlpha:c})]}case"FusedDepthwiseConv2dNative":{const{stride:r,pad:i,dataFormat:o,dilations:u,biasArg:p,preluArg:m,activationFunc:l,leakyreluAlpha:c}=gt(a,e,t);return[s.fused.depthwiseConv2d({x:n("x",a,e,t),filter:n("filter",a,e,t),strides:[r[1],r[2]],pad:i,dataFormat:o,dilations:[u[1],u[2]],bias:p,activation:l,preluActivationWeights:m,leakyreluAlpha:c})]}case"Conv2DBackpropInput":case"Conv2dTranspose":{const r=n("outputShape",a,e,t),i=n("strides",a,e,t),o=ge(a,e,t);return[s.conv2dTranspose(n("x",a,e,t),n("filter",a,e,t),r,[i[1],i[2]],o)]}case"DepthwiseConv2dNative":case"DepthwiseConv2d":{const r=n("strides",a,e,t),i=ge(a,e,t),o=n("dilations",a,e,t),u=n("dataFormat",a,e,t).toUpperCase();return[s.depthwiseConv2d(n("input",a,e,t),n("filter",a,e,t),[r[1],r[2]],i,u,[o[1],o[2]])]}case"Conv3D":{const r=n("strides",a,e,t),i=n("pad",a,e,t),o=n("dataFormat",a,e,t).toUpperCase(),u=n("dilations",a,e,t);return[s.conv3d(n("x",a,e,t),n("filter",a,e,t),[r[1],r[2],r[3]],i,o,[u[1],u[2],u[3]])]}case"AvgPool":{const r=n("strides",a,e,t),i=n("pad",a,e,t),o=n("kernelSize",a,e,t);return[s.avgPool(n("x",a,e,t),[o[1],o[2]],[r[1],r[2]],i)]}case"MaxPool":{const r=n("strides",a,e,t),i=n("pad",a,e,t),o=n("kernelSize",a,e,t);return[s.maxPool(n("x",a,e,t),[o[1],o[2]],[r[1],r[2]],i)]}case"MaxPoolWithArgmax":{const r=n("strides",a,e,t),i=n("pad",a,e,t),o=n("kernelSize",a,e,t),u=n("includeBatchInIndex",a,e,t),{result:p,indexes:m}=s.maxPoolWithArgmax(n("x",a,e,t),[o[1],o[2]],[r[1],r[2]],i,u);return[p,m]}case"AvgPool3D":{const r=n("strides",a,e,t),i=n("pad",a,e,t),o=n("kernelSize",a,e,t);return[s.avgPool3d(n("x",a,e,t),[o[1],o[2],o[3]],[r[1],r[2],r[3]],i)]}case"MaxPool3D":{const r=n("strides",a,e,t),i=n("pad",a,e,t),o=n("kernelSize",a,e,t);return[s.maxPool3d(n("x",a,e,t),[o[1],o[2],o[3]],[r[1],r[2],r[3]],i)]}case"Dilation2D":{const r=n("strides",a,e,t),i=n("pad",a,e,t),o=n("dilations",a,e,t),u=r[1],p=r[2],m=o[1],l=o[2];return[s.dilation2d(n("x",a,e,t),n("filter",a,e,t),[u,p],i,[m,l],"NHWC")]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const yu=(a,e,t,s=P)=>{switch(a.op){case"Fill":{const r=n("shape",a,e,t),i=n("dtype",a,e,t),o=n("value",a,e,t);return[s.fill(r,o,i)]}case"LinSpace":{const r=n("start",a,e,t),i=n("stop",a,e,t),o=n("num",a,e,t);return[s.linspace(r,i,o)]}case"Multinomial":{const r=n("logits",a,e,t),i=n("numSamples",a,e,t),o=n("seed",a,e,t);return[s.multinomial(r,i,o)]}case"OneHot":{const r=n("indices",a,e,t),i=n("depth",a,e,t),o=n("onValue",a,e,t),u=n("offValue",a,e,t),p=n("dtype",a,e,t);return[s.oneHot(r,i,o,u,p)]}case"Ones":return[s.ones(n("shape",a,e,t),n("dtype",a,e,t))];case"OnesLike":return[s.onesLike(n("x",a,e,t))];case"RandomStandardNormal":return[s.randomStandardNormal(n("shape",a,e,t),n("dtype",a,e,t),n("seed",a,e,t))];case"RandomUniform":return[s.randomUniform(n("shape",a,e,t),n("minval",a,e,t),n("maxval",a,e,t),n("dtype",a,e,t))];case"Range":{const r=n("start",a,e,t),i=n("stop",a,e,t),o=n("step",a,e,t);return[s.range(r,i,o,n("dtype",a,e,t))]}case"TruncatedNormal":{const r=n("shape",a,e,t),i=n("mean",a,e,t),o=n("stdDev",a,e,t),u=n("seed",a,e,t);return[s.truncatedNormal(r,i,o,n("dtype",a,e,t),u)]}case"Zeros":return[s.zeros(n("shape",a,e,t),n("dtype",a,e,t))];case"ZerosLike":return[s.zerosLike(n("x",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function Ce(a,e,t){const s=n("boxes",a,e,t),r=n("scores",a,e,t),i=n("maxOutputSize",a,e,t),o=n("iouThreshold",a,e,t),u=n("scoreThreshold",a,e,t),p=n("softNmsSigma",a,e,t);return{boxes:s,scores:r,maxOutputSize:i,iouThreshold:o,scoreThreshold:u,softNmsSigma:p}}const gu=async(a,e,t,s,r=P)=>{switch(a.op){case"NonMaxSuppressionV5":{const{boxes:i,scores:o,maxOutputSize:u,iouThreshold:p,scoreThreshold:m,softNmsSigma:l}=Ce(a,e,t),c=await r.image.nonMaxSuppressionWithScoreAsync(i,o,u,p,m,l);return[c.selectedIndices,c.selectedScores]}case"NonMaxSuppressionV4":{const{boxes:i,scores:o,maxOutputSize:u,iouThreshold:p,scoreThreshold:m}=Ce(a,e,t),l=n("padToMaxOutputSize",a,e,t),c=await r.image.nonMaxSuppressionPaddedAsync(i,o,u,p,m,l);return[c.selectedIndices,c.validOutputs]}case"NonMaxSuppressionV3":case"NonMaxSuppressionV2":{const{boxes:i,scores:o,maxOutputSize:u,iouThreshold:p,scoreThreshold:m}=Ce(a,e,t);return[await r.image.nonMaxSuppressionAsync(i,o,u,p,m)]}case"Where":{const i=r.cast(n("condition",a,e,t),"bool"),o=[await r.whereAsync(i)];return i.dispose(),o}case"ListDiff":return r.setdiff1dAsync(n("x",a,e,t),n("y",a,e,t));default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const Nu=(a,e,t,s=P)=>{switch(a.op){case"LowerBound":{const r=n("sortedSequence",a,e,t),i=n("values",a,e,t);return[s.lowerBound(r,i)]}case"TopKV2":{const r=n("x",a,e,t),i=n("k",a,e,t),o=n("sorted",a,e,t),u=s.topk(r,i,o);return[u.values,u.indices]}case"UpperBound":{const r=n("sortedSequence",a,e,t),i=n("values",a,e,t);return[s.upperBound(r,i)]}case"Unique":{const r=n("x",a,e,t),i=s.unique(r);return[i.values,i.indices]}case"UniqueV2":{const r=n("x",a,e,t),i=n("axis",a,e,t),o=s.unique(r,i);return[o.values,o.indices]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const bu=(a,e,t,s=P)=>{switch(a.op){case"Const":return e[a.name];case"PlaceholderWithDefault":const r=n("default",a,e,t);return[L(a.name,e,t)||r];case"Placeholder":return[L(a.name,e,t)];case"Identity":case"StopGradient":case"FakeQuantWithMinMaxVars":{const l=n("x",a,e,t);return[U(l)]}case"IdentityN":return n("x",a,e,t).map(l=>U(l));case"Snapshot":const i=n("x",a,e,t);return[U(i)];case"Shape":return[s.tensor1d(n("x",a,e,t).shape,"int32")];case"ShapeN":return n("x",a,e,t).map(l=>s.tensor1d(l.shape));case"Size":return[s.scalar(n("x",a,e,t).size,"int32")];case"Rank":return[s.scalar(n("x",a,e,t).rank,"int32")];case"NoOp":return[s.scalar(1)];case"Print":const o=n("x",a,e,t),u=n("data",a,e,t),p=n("message",a,e,t),m=n("summarize",a,e,t);console.warn("The graph has a tf.print() operation,usually used for debugging, which slows down performance."),console.log(p);for(let l=0;l<u.length;l++)console.log(Array.prototype.slice.call(u[l].dataSync()).slice(0,m));return[o];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */class Tu{get id(){return this.handle.id}constructor(e,t){this.keyDType=e,this.valueDType=t,this.handle=K(0),this.tensorMap=new Map,G(this.handle)}clearAndClose(){this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return K(this.size(),"int32")}async import(e,t){this.checkKeyAndValueTensor(e,t);const s=await e.data();return this.tensorMap.forEach(r=>r.dispose()),this.tensorMap.clear(),R(()=>{const r=ce(t),i=s.length,o=r.length;S(i===o,()=>`The number of elements doesn't match, keys has ${i} elements, the values has ${o} elements.`);for(let u=0;u<i;u++){const p=s[u],m=r[u];G(m),this.tensorMap.set(p,m)}return this.handle})}async find(e,t){this.checkKeyAndValueTensor(e,t);const s=await e.data();return R(()=>{const r=[];for(let i=0;i<s.length;i++){const o=s[i],u=this.findWithDefault(o,t);r.push(u)}return re(r)})}findWithDefault(e,t){const s=this.tensorMap.get(e);return s??t}checkKeyAndValueTensor(e,t){if(e.dtype!==this.keyDType)throw new Error(`Expect key dtype ${this.keyDType}, but got ${e.dtype}`);if(t.dtype!==this.valueDType)throw new Error(`Expect value dtype ${this.valueDType}, but got ${t.dtype}`)}}/**
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
 */const wu=async(a,e,t,s)=>{switch(a.op){case"HashTable":case"HashTableV2":{const r=s.getHashTableHandleByName(a.name);if(r!=null)return[r];{const i=n("keyDType",a,e,t),o=n("valueDType",a,e,t),u=new Tu(i,o);return s.addHashTable(a.name,u),[u.handle]}}case"InitializeTable":case"InitializeTableV2":case"LookupTableImport":case"LookupTableImportV2":{const r=n("tableHandle",a,e,t,s),i=n("keys",a,e,t),o=n("values",a,e,t);return[await s.getHashTableById(r.id).import(i,o)]}case"LookupTableFind":case"LookupTableFindV2":{const r=n("tableHandle",a,e,t,s),i=n("keys",a,e,t),o=n("defaultValue",a,e,t);return[await s.getHashTableById(r.id).find(i,o)]}case"LookupTableSize":case"LookupTableSizeV2":{const r=n("tableHandle",a,e,t,s);return[s.getHashTableById(r.id).tensorSize()]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const Su=(a,e,t,s=P)=>{switch(a.op){case"ResizeBilinear":{const r=n("images",a,e,t),i=n("size",a,e,t),o=n("alignCorners",a,e,t),u=n("halfPixelCenters",a,e,t);return[s.image.resizeBilinear(r,[i[0],i[1]],o,u)]}case"ResizeNearestNeighbor":{const r=n("images",a,e,t),i=n("size",a,e,t),o=n("alignCorners",a,e,t),u=n("halfPixelCenters",a,e,t);return[s.image.resizeNearestNeighbor(r,[i[0],i[1]],o,u)]}case"CropAndResize":{const r=n("image",a,e,t),i=n("boxes",a,e,t),o=n("boxInd",a,e,t),u=n("cropSize",a,e,t),p=n("method",a,e,t),m=n("extrapolationValue",a,e,t);return[s.image.cropAndResize(r,i,o,u,p,m)]}case"ImageProjectiveTransformV3":{const r=n("images",a,e,t),i=n("transforms",a,e,t),o=n("outputShape",a,e,t),u=n("fillValue",a,e,t),p=n("interpolation",a,e,t),m=n("fillMode",a,e,t);return[s.image.transform(r,i,p.toLowerCase(),m.toLowerCase(),u,o)]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const _u=(a,e,t,s=P)=>{switch(a.op){case"Equal":return[s.equal(n("a",a,e,t),n("b",a,e,t))];case"NotEqual":return[s.notEqual(n("a",a,e,t),n("b",a,e,t))];case"Greater":return[s.greater(n("a",a,e,t),n("b",a,e,t))];case"GreaterEqual":return[s.greaterEqual(n("a",a,e,t),n("b",a,e,t))];case"Less":return[s.less(n("a",a,e,t),n("b",a,e,t))];case"LessEqual":return[s.lessEqual(n("a",a,e,t),n("b",a,e,t))];case"LogicalAnd":return[s.logicalAnd(n("a",a,e,t),n("b",a,e,t))];case"LogicalNot":return[s.logicalNot(n("a",a,e,t))];case"LogicalOr":return[s.logicalOr(n("a",a,e,t),n("b",a,e,t))];case"Select":case"SelectV2":return[s.where(n("condition",a,e,t),n("a",a,e,t),n("b",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const vu=(a,e,t,s=P)=>{switch(a.op){case"BatchMatMul":case"BatchMatMulV2":case"MatMul":return[s.matMul(n("a",a,e,t),n("b",a,e,t),n("transposeA",a,e,t),n("transposeB",a,e,t))];case"Einsum":return[s.einsum(n("equation",a,e,t),...n("tensors",a,e,t))];case"Transpose":return[s.transpose(n("x",a,e,t),n("perm",a,e,t))];case"_FusedMatMul":const[r,i]=n("fusedOps",a,e,t),o=r==="biasadd",u=i==="prelu",p=n("numArgs",a,e,t),m=n("leakyreluAlpha",a,e,t);if(o){if(u&&p!==2)throw new Error("Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!u&&p!==1)throw new Error("Fused MatMul with BiasAdd must have one extra argument: bias.")}const[l,c]=n("args",a,e,t);return[s.fused.matMul({a:n("a",a,e,t),b:n("b",a,e,t),transposeA:n("transposeA",a,e,t),transposeB:n("transposeB",a,e,t),bias:l,activation:i,preluActivationWeights:c,leakyreluAlpha:m})];case"MatrixBandPart":return[s.linalg.bandPart(n("a",a,e,t),n("numLower",a,e,t),n("numUpper",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const Ou=(a,e,t,s=P)=>{switch(a.op){case"EuclideanNorm":return[s.euclideanNorm(n("x",a,e,t),n("axis",a,e,t),n("keepDims",a,e,t))];case"FusedBatchNorm":case"FusedBatchNormV2":return[s.batchNorm(n("x",a,e,t),n("mean",a,e,t),n("variance",a,e,t),n("offset",a,e,t),n("scale",a,e,t),n("epsilon",a,e,t))];case"FusedBatchNormV3":return[s.batchNorm(n("x",a,e,t),n("mean",a,e,t),n("variance",a,e,t),n("offset",a,e,t),n("scale",a,e,t),n("epsilon",a,e,t))];case"LRN":return[s.localResponseNormalization(n("x",a,e,t),n("radius",a,e,t),n("bias",a,e,t),n("alpha",a,e,t),n("beta",a,e,t))];case"Softmax":return[s.softmax(n("x",a,e,t))];case"LogSoftmax":return[s.logSoftmax(n("x",a,e,t))];case"SparseToDense":return[s.sparseToDense(n("sparseIndices",a,e,t),n("outputShape",a,e,t),n("sparseValues",a,e,t),n("defaultValue",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const Eu=(a,e,t,s=P)=>{switch(a.op){case"RaggedGather":{const{outputNestedSplits:r,outputDenseValues:i}=s.raggedGather(n("paramsNestedSplits",a,e,t),n("paramsDenseValues",a,e,t),n("indices",a,e,t),n("outputRaggedRank",a,e,t));return r.concat(i)}case"RaggedRange":{const{rtNestedSplits:r,rtDenseValues:i}=s.raggedRange(n("starts",a,e,t),n("limits",a,e,t),n("splits",a,e,t));return[r,i]}case"RaggedTensorToTensor":return[s.raggedTensorToTensor(n("shape",a,e,t),n("values",a,e,t),n("defaultValue",a,e,t),n("rowPartitionTensors",a,e,t),n("rowPartitionTypes",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const Iu=(a,e,t,s=P)=>{switch(a.op){case"Max":{const u=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.max(n("x",a,e,t),u,p)]}case"Mean":{const u=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.mean(n("x",a,e,t),u,p)]}case"Min":{const u=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.min(n("x",a,e,t),u,p)]}case"Sum":{const u=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.sum(n("x",a,e,t),u,p)]}case"All":{const u=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.all(n("x",a,e,t),u,p)]}case"Any":{const u=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.any(n("x",a,e,t),u,p)]}case"ArgMax":{const u=n("axis",a,e,t);return[s.argMax(n("x",a,e,t),u)]}case"ArgMin":{const u=n("axis",a,e,t);return[s.argMin(n("x",a,e,t),u)]}case"Prod":{const u=n("axis",a,e,t),p=n("keepDims",a,e,t);return[s.prod(n("x",a,e,t),u,p)]}case"Cumprod":{const u=n("axis",a,e,t),p=n("exclusive",a,e,t),m=n("reverse",a,e,t);return[s.cumprod(n("x",a,e,t),u,p,m)]}case"Cumsum":{const u=n("axis",a,e,t),p=n("exclusive",a,e,t),m=n("reverse",a,e,t);return[s.cumsum(n("x",a,e,t),u,p,m)]}case"Bincount":const r=n("x",a,e,t),i=n("weights",a,e,t),o=n("size",a,e,t);return[s.bincount(r,i,o)];case"DenseBincount":{const u=n("x",a,e,t),p=n("weights",a,e,t),m=n("size",a,e,t),l=n("binaryOutput",a,e,t);return[s.denseBincount(u,p,m,l)]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const ku=(a,e,t,s=P)=>{switch(a.op){case"ConcatV2":case"Concat":{const r=n("n",a,e,t),i=n("axis",a,e,t);let o=n("tensors",a,e,t);return o=o.slice(0,r),[s.concat(o,i)]}case"Gather":{const r=n("x",a,e,t),i=n("indices",a,e,t);return[s.gather(r,s.cast(i,"int32"),0)]}case"GatherV2":{const r=n("axis",a,e,t),i=n("batchDims",a,e,t),o=n("x",a,e,t),u=n("indices",a,e,t);return[s.gather(o,s.cast(u,"int32"),r,i)]}case"Reverse":{const r=n("dims",a,e,t),i=[];for(let u=0;u<r.length;u++)r[u]&&i.push(u);const o=n("x",a,e,t);return[s.reverse(o,i)]}case"ReverseV2":{const r=n("axis",a,e,t),i=n("x",a,e,t);return[s.reverse(i,r)]}case"Slice":{const r=n("begin",a,e,t),i=n("size",a,e,t);return[s.slice(n("x",a,e,t),r,i)]}case"StridedSlice":{const r=n("begin",a,e,t),i=n("end",a,e,t),o=n("strides",a,e,t),u=n("beginMask",a,e,t),p=n("endMask",a,e,t),m=n("ellipsisMask",a,e,t),l=n("newAxisMask",a,e,t),c=n("shrinkAxisMask",a,e,t),d=n("x",a,e,t);return[s.stridedSlice(d,r,i,o,u,p,m,l,c)]}case"Pack":return R(()=>{const r=n("axis",a,e,t),i=n("tensors",a,e,t),o=i[0].shape,u=s.squeeze(i[0]).shape,p=i.map(m=>{const l=Ne(m.shape,o);if(!l&&!Ne(s.squeeze(m).shape,u))throw new Error("the input tensors shape does not match");return l?m:s.reshape(m,o)});return[s.stack(p,r)]});case"Unpack":{const r=n("axis",a,e,t),i=n("tensor",a,e,t);return s.unstack(i,r)}case"Tile":{const r=n("reps",a,e,t);return[s.tile(n("x",a,e,t),r)]}case"Split":case"SplitV":{const r=n("axis",a,e,t),i=n("numOrSizeSplits",a,e,t),o=n("x",a,e,t);return s.split(o,i,r)}case"ScatterNd":{const r=n("indices",a,e,t),i=n("values",a,e,t),o=n("shape",a,e,t);return[s.scatterND(r,i,o)]}case"GatherNd":{const r=n("x",a,e,t),i=n("indices",a,e,t);return[s.gatherND(r,i)]}case"SparseToDense":{const r=n("sparseIndices",a,e,t),i=n("outputShape",a,e,t),o=n("sparseValues",a,e,t),u=n("defaultValue",a,e,t);return[s.sparseToDense(r,o,i,o.dtype===u.dtype?u:s.cast(u,o.dtype))]}case"TensorScatterUpdate":{const r=n("indices",a,e,t),i=n("values",a,e,t),o=n("tensor",a,e,t);return[s.tensorScatterUpdate(o,r,i)]}default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const Au=(a,e,t,s=P)=>{switch(a.op){case"SparseFillEmptyRows":{const{outputIndices:r,outputValues:i,emptyRowIndicator:o,reverseIndexMap:u}=s.sparse.sparseFillEmptyRows(n("indices",a,e,t),n("values",a,e,t),n("denseShape",a,e,t),n("defaultValue",a,e,t));return[r,i,o,u]}case"SparseReshape":{const{outputIndices:r,outputShape:i}=s.sparse.sparseReshape(n("inputIndices",a,e,t),n("inputShape",a,e,t),n("newShape",a,e,t));return[r,i]}case"SparseSegmentMean":return[s.sparse.sparseSegmentMean(n("data",a,e,t),n("indices",a,e,t),n("segmentIds",a,e,t))];case"SparseSegmentSum":return[s.sparse.sparseSegmentSum(n("data",a,e,t),n("indices",a,e,t),n("segmentIds",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const Du=(a,e,t,s=P)=>{switch(a.op){case"FFT":return[s.fft(n("x",a,e,t))];case"IFFT":return[s.ifft(n("x",a,e,t))];case"RFFT":return[s.rfft(n("x",a,e,t))];case"IRFFT":return[s.irfft(n("x",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
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
 */const $u=(a,e,t,s=P)=>{switch(a.op){case"StaticRegexReplace":return[s.string.staticRegexReplace(n("input",a,e,t),n("pattern",a,e,t),n("rewrite",a,e,t),n("replaceGlobal",a,e,t))];case"StringNGrams":{const{nGrams:r,nGramsSplits:i}=s.string.stringNGrams(n("data",a,e,t),n("dataSplits",a,e,t),n("separator",a,e,t),n("nGramWidths",a,e,t),n("leftPad",a,e,t),n("rightPad",a,e,t),n("padWidth",a,e,t),n("preserveShortSequences",a,e,t));return[r,i]}case"StringSplit":{const{indices:r,values:i,shape:o}=s.string.stringSplit(n("input",a,e,t),n("delimiter",a,e,t),n("skipEmpty",a,e,t));return[r,i,o]}case"StringToHashBucketFast":return[s.string.stringToHashBucketFast(n("input",a,e,t),n("numBuckets",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const Cu=(a,e,t,s=P)=>{switch(a.op){case"Cast":return[s.cast(n("x",a,e,t),n("dtype",a,e,t))];case"ExpandDims":{const r=n("axis",a,e,t);return[s.expandDims(n("x",a,e,t),r)]}case"Squeeze":{const r=n("axis",a,e,t);return[s.squeeze(n("x",a,e,t),r)]}case"Reshape":return[s.reshape(n("x",a,e,t),n("shape",a,e,t))];case"MirrorPad":return[s.mirrorPad(n("x",a,e,t),n("padding",a,e,t),n("mode",a,e,t))];case"PadV2":case"Pad":return[s.pad(n("x",a,e,t),n("padding",a,e,t),n("constantValue",a,e,t))];case"SpaceToBatchND":{const r=n("blockShape",a,e,t),i=n("paddings",a,e,t);return[s.spaceToBatchND(n("x",a,e,t),r,i)]}case"BatchToSpaceND":{const r=n("blockShape",a,e,t),i=n("crops",a,e,t);return[s.batchToSpaceND(n("x",a,e,t),r,i)]}case"DepthToSpace":{const r=n("blockSize",a,e,t),i=n("dataFormat",a,e,t).toUpperCase();return[s.depthToSpace(n("x",a,e,t),r,i)]}case"BroadcastTo":return[s.broadcastTo(n("x",a,e,t),n("shape",a,e,t))];case"BroadcastArgs":return[s.broadcastArgs(n("s0",a,e,t),n("s1",a,e,t))];default:throw TypeError(`Node type ${a.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */function Nt(a,e,t,s,r=R){const i=((o,u,p)=>{switch(o.category){case"arithmetic":return r(()=>ou(o,u,p));case"basic_math":return r(()=>uu(o,u,p));case"control":return hu(o,u,p);case"convolution":return r(()=>fu(o,u,p));case"creation":return r(()=>yu(o,u,p));case"dynamic":return gu(o,u,p);case"evaluation":return r(()=>Nu(o,u,p));case"image":return r(()=>Su(o,u,p));case"graph":return r(()=>bu(o,u,p));case"logical":return r(()=>_u(o,u,p));case"matrices":return r(()=>vu(o,u,p));case"normalization":return r(()=>Ou(o,u,p));case"ragged":return r(()=>Eu(o,u,p));case"reduction":return r(()=>Iu(o,u,p));case"slice_join":return r(()=>ku(o,u,p));case"sparse":return r(()=>Au(o,u,p));case"spectral":return r(()=>Du(o,u,p));case"string":return r(()=>$u(o,u,p));case"transformation":return r(()=>Cu(o,u,p));case"hash_table":return wu(o,u,p,s);case"custom":const m=Ut(o.op);if(m&&m.customExecutor)return m.customExecutor(new iu(o,u,p));throw TypeError(`Custom op ${o.op} is not registered.`);default:throw TypeError(`Unknown op '${o.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(a,e,t);return Te(i)?i.then(o=>[].concat(o)):[].concat(i)}class bt{constructor(e={},t={},s={},r={},i){this.weightMap=e,this.tensorArrayMap=t,this.tensorListMap=s,this.functionMap=r,this.parseNodeNameCache=i,this.rootContext={id:0,frameName:"",iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(e,t){return{id:e,frameName:t,iterationId:0}}set currentContext(e){this.contexts!==e&&(this.contexts=e,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){const e=[];for(let t=0;t<this.contexts.length-1;t++){const s=this.contexts.slice(0,this.contexts.length-t);e.push(this.contextIdforContexts(s))}e.push(""),this._currentContextIds=e}contextIdforContexts(e){return e?e.map(t=>t.id===0&&t.iterationId===0?"":`${t.frameName}-${t.iterationId}`).join("/"):""}enterFrame(e){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,e)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw new Error("Cannot exit frame, the context is empty")}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;const e=Object.assign({},this.contexts[this.contexts.length-1]);e.iterationId+=1,e.id=this.lastId,this.contexts.splice(-1,1,e),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw new Error("Cannot increase frame iteration, the context is empty")}getWeight(e){return this.weightMap[e]}addTensorArray(e){this.tensorArrayMap[e.id]=e}getTensorArray(e){return this.tensorArrayMap[e]}addTensorList(e){this.tensorListMap[e.id]=e}getTensorList(e){return this.tensorListMap[e]}dispose(e){for(const t in this.tensorArrayMap)this.tensorArrayMap[t].clearAndClose(e);for(const t in this.tensorListMap)this.tensorListMap[t].clearAndClose(e)}}/**
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
 */function Tt(a,e,t,s){const r=new Set,i=[];let o=null,u=null;const p=new Set,m=new Set(Object.keys(a).map(d=>F(d)[0]));s=s||[];const l=new Set(s.map(d=>F(d.name)[0])),c=[...e];for(;c.length>0;){const d=c.pop();if((X(d)||Ru(d)||Bu(d))&&o==null&&(o=d,u=o.children.map(f=>f.name).filter(f=>r.has(f))),r.add(d.name),t[d.name]==null&&!m.has(d.name)&&!l.has(d.name)){if(d.inputs.length===0){i.push(d.name);continue}d.inputs.forEach(f=>{p.has(f.name)||(p.add(f.name),c.push(f))})}}return{inputs:a,outputs:e,usedNodes:r,missingInputs:i,dynamicNode:o,syncInputs:u}}function zu(a,e){const{usedNodes:t,inputs:s}=e,r=Object.keys(s).map(h=>F(h)[0]).map(h=>a.nodes[h]),i=a.initNodes||[],o=h=>t.has(typeof h=="string"?h:h.name);function u(h){return[...new Map(h.map(N=>[N.name,N])).values()]}const p=u([...r,...a.weights,...i]).filter(o),m=u([...p,...Object.values(a.nodes)]).filter(o),l=new Map(m.map(h=>[h.name,h])),c={};for(const h of m){c[h.name]=c[h.name]||0;for(const N of h.children)o(N)||(c[N.name]=Number.POSITIVE_INFINITY),c[N.name]=(c[N.name]||0)+1}const d=Object.entries(c).filter(([,h])=>h===0).map(([h])=>h),f=[...d];for(;d.length>0;){const h=d.pop(),N=l.get(h);for(const _ of N.children.filter(o))--c[_.name]===0&&(f.push(_.name),d.push(_.name))}const g=f.map(h=>l.get(h)),y=Lu(g,p);return Pu(y,p),y}function Lu(a,e){const t=new Map(a.map(o=>[o.name,o])),s=e.map(o=>o.name),r=new Set(s);for(;s.length>0;){const o=s.pop(),u=t.get(o);for(const p of u.children)!t.has(p.name)||r.has(p.name)||(r.add(p.name),s.push(p.name))}return a.filter(o=>r.has(o.name))}class he extends Error{constructor(e){super(`NodesExecutionOrderError: ${e}`)}}function Pu(a,e){const t=new Map(a.map((u,p)=>[u.name,p])),s=new Set(e.map(u=>u.name)),r=u=>s.has(typeof u=="string"?u:u.name),i=new Set(a.map(u=>u.name)),o=u=>i.has(typeof u=="string"?u:u.name);for(const u of a){for(const p of u.children.filter(o)){if(!t.has(p.name))throw new he(`Child ${p.name} of node ${u.name} is unreachable.`);if(t.get(u.name)>t.get(p.name))throw new he(`Node ${u.name} is scheduled to run after its child ${p.name}.`)}if(!r(u))for(const p of u.inputs){if(!t.has(p.name))throw new he(`Input ${p.name} of node ${u.name} is unreachable.`);if(t.get(p.name)>t.get(u.name))throw new he(`Node ${u.name} is scheduled to run before its input ${p.name}.`)}}}function Vu(a){const e=new Map(a.map((u,p)=>[u.name,p])),t=Number.MAX_SAFE_INTEGER,s=a.map((u,p)=>X(u)?t:p),r=u=>{const p=s[e.get(u.name)];return p??-1},i=a.map((u,p)=>u.children.map(r).reduce((m,l)=>Math.max(m,l),s[p])),o=new Map;for(let u=0;u<a.length;++u){const p=i[u];if(p===t)continue;const m=a[u],l=a[p];o.has(l.name)||o.set(l.name,[]),o.get(l.name).push(m)}return o}const Fu=new Set(["Switch","Merge","Enter","Exit","NextIteration","StatelessIf","StatelessWhile","if","While"]),ju=new Set(["NonMaxSuppressionV2","NonMaxSuppressionV3","NonMaxSuppressionV5","Where"]),xu=new Set(["HashTable","HashTableV2","LookupTableImport","LookupTableImportV2","LookupTableFind","LookupTableFindV2","LookupTableSize","LookupTableSizeV2"]);function X(a){return Fu.has(a.op)}function Ru(a){return ju.has(a.op)}function Bu(a){return xu.has(a.op)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */class Se{get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(e){const t=Object.keys(e).map(s=>e[s].map(r=>r.id));this._weightIds=[].concat(...t),this._weightMap=e}set resourceManager(e){this._resourceManager=e}get inputs(){return this._inputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(e=>e.signatureKey||e.name)}get outputNodes(){return this._outputs.map(e=>{const t=e.signatureKey||e.name;return e.defaultOutput?`${t}:${e.defaultOutput}`:t})}get functions(){return Object.keys(this._functions).reduce((e,t)=>(e[t]=this._functions[t].signature,e),{})}constructor(e,t){this.graph=e,this.parent=t,this.compiledMap=new Map,this.parseNodeNameCache=new Map,this._weightMap={},this.SEPARATOR=",",this._functions={},this._functionExecutorMap={},this.keepIntermediateTensors=!1,this._outputs=e.outputs,this._inputs=e.inputs,this._initNodes=e.initNodes,this._signature=e.signature,this._functions=e.functions,e.functions!=null&&Object.keys(e.functions).forEach(s=>{this._functionExecutorMap[s]=new Se(e.functions[s],this)})}getCompilationKey(e,t){const s=e.map(i=>i.name).sort(),r=t.map(i=>i.name).sort();return s.join(this.SEPARATOR)+"--"+r.join(this.SEPARATOR)}compile(e,t){const s=Tt(e,t,this.weightMap,this._initNodes),{missingInputs:r,dynamicNode:i,syncInputs:o}=s;if(i!=null)throw new Error(`This execution contains the node '${i.name}', which has the dynamic op '${i.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${o}]`);if(r.length>0){const m=t.map(c=>c.name),l=Object.keys(e);throw new Error(`Cannot compute the outputs [${m}] from the provided inputs [${l}]. Missing the following inputs: [${r}]`)}const u=zu(this.graph,s),p=Vu(u);return{orderedNodes:u,nodeLiveUntilMap:p}}cloneAndKeepTensor(e){if(e==null)return null;const t=e.clone();return G(t),t}cloneTensorList(e){return e?e.map(s=>this.cloneAndKeepTensor(s)):null}cloneTensorMap(e){return Object.fromEntries(Object.entries(e).map(([t,s])=>[t,this.cloneTensorList(s)]))}execute(e,t){this.disposeIntermediateTensors(),e=this.mapInputs(e);const s=Object.keys(e).sort();this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t);const r=s.map(d=>this.graph.nodes[F(d)[0]]),i=t.map(d=>F(d)[0]),o=new Set(i);let u=i.map(d=>this.graph.nodes[d]);u.length===0&&(u=this._outputs);const p=this.getCompilationKey(r,u);let m=this.compiledMap.get(p);m==null&&(m=this.compile(e,u),this.compiledMap.set(p,m));try{this.keepIntermediateTensors=Q().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(d){this.keepIntermediateTensors=!1,console.warn(d.message)}const l={},c={};return R(()=>{const d=new bt(this.weightMap,l,c,this.functionExecutorMap,this.parseNodeNameCache),f=Object.assign({},this.weightMap);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap)),Object.keys(e).forEach(N=>{const[_,A]=F(N,d),T=[];T[A]=e[N],f[_]=T,this.keepIntermediateTensors&&(this.clonedTensorsMap[_]=this.cloneTensorList(T))});const g=this.getFrozenTensorIds(f),{orderedNodes:y,nodeLiveUntilMap:h}=m;for(const N of y){if(f[N.name])continue;const _=Nt(N,f,d,this._resourceManager);if(Te(_))throw new Error(`The execution of the op '${N.op}' returned a promise. Please use model.executeAsync() instead.`);f[N.name]=_,this.keepIntermediateTensors&&(this.clonedTensorsMap[N.name]=this.cloneTensorList(_)),this.checkTensorForDisposalWithNodeLiveUntilInfo(N,f,d,g,o,h.get(N.name))}return this.parent==null&&d.dispose(g),t.map(N=>L(N,f,d))})}getFrozenTensorIds(e){const t=[].concat.apply([],Object.keys(e).map(s=>e[s]).map(s=>s.map(r=>r.id)));return new Set(t)}checkTensorForDisposal(e,t,s,r,i,o,u){if(!(X(t)||o.has(e))){for(const p of s[e])p!=null&&(u[p.id]=(u[p.id]||0)+t.children.length);for(const p of t.inputs){if(X(p))continue;const m=dt(p.name,s,r);if(m!=null)for(const l of m){if(!l||l.kept||i.has(l.id))continue;const c=u[l.id];c===1?(l.dispose(),delete u[l.id]):c!=null&&u[l.id]--}}}}checkTensorForDisposalWithNodeLiveUntilInfo(e,t,s,r,i,o){function u(p){return X(p)||i.has(p.name)}if(!(X(e)||o==null))for(const p of o){if(u(p))continue;const m=dt(p.name,t,s);for(const l of m)!l||l.kept||r.has(l.id)||l.dispose()}}async executeAsync(e,t){return this._executeAsync(e,t)}disposeIntermediateTensors(){this.clonedTensorsMap&&(Object.values(this.clonedTensorsMap).forEach(e=>{for(const t of e)t&&!t.isDisposed&&t.dispose()}),this.clonedTensorsMap=null)}getIntermediateTensors(){return this.clonedTensorsMap}async _executeAsync(e,t,s=!1,r={},i={}){this.disposeIntermediateTensors(),s||(e=this.mapInputs(e),this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t));try{this.keepIntermediateTensors=Q().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(d){this.keepIntermediateTensors=!1,console.warn(d.message)}const o=new bt(this.weightMap,r,i,this.functionExecutorMap,this.parseNodeNameCache);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap));const u=await this.executeWithControlFlow(e,o,t,s),p=t.map(d=>L(d,u,o)),m=p.map(d=>d.id),l=Object.keys(e).map(d=>e[d].id),c=new Set([...m,...l,...this.weightIds]);return Object.values(u).forEach(d=>{d.forEach(f=>{f&&!f.isDisposed&&!c.has(f.id)&&f.dispose()})}),this.parent==null&&o.dispose(c),p}async executeFunctionAsync(e,t,s){const r=e.reduce((i,o,u)=>(i[this.inputs[u].name]=o,i),{});return this._executeAsync(r,this.outputNodes,!0,t,s)}async executeWithControlFlow(e,t,s,r){const i=Object.keys(e),o=i.map(T=>this.graph.nodes[F(T)[0]]),u=s.map(T=>F(T)[0]),p=new Set(u);let m=u.map(T=>this.graph.nodes[T]);m.length===0&&(m=this._outputs);const{usedNodes:l,missingInputs:c,dynamicNode:d,syncInputs:f}=Tt(e,m,this.weightMap,this._initNodes),g=[...o,...this.graph.weights,...this._initNodes||[]].map(T=>({node:T,contexts:t.currentContext})),y=Object.assign({},this.weightMap);Object.keys(e).forEach(T=>{const[I,$]=F(T),w=[];w[$]=e[T],y[I]=w});const h={},N=this.getFrozenTensorIds(y),_={};for(;g.length>0;){const T=this.processStack(o,g,t,y,_,N,p,h,l);await Promise.all(T)}d==null&&!r&&console.warn("This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.");const A=m.filter(T=>!X(T)&&!L(T.name,y,t)).map(T=>T.name);if(A.length>0){let T="";throw d!=null&&(T=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${f}]`),new Error(`Cannot compute the outputs [${A}] from the provided inputs [${i}]. Consider providing the following inputs: [${c}]. ${T}`)}return y}processStack(e,t,s,r,i,o,u,p,m){const l=[];for(;t.length>0;){const c=t.pop();s.currentContext=c.contexts;let d="";if(c.node.op==="Enter"&&n("isConstant",c.node,r,s)&&([d]=W(c.node.name,s)),r[c.node.name]==null){const f=Nt(c.node,r,s,this._resourceManager);d||([d]=W(c.node.name,s));const g=s.currentContext;Te(f)?l.push(f.then(y=>(r[d]=y,this.keepIntermediateTensors&&(this.clonedTensorsMap[d]=this.cloneTensorList(y)),s.currentContext=g,this.checkTensorForDisposal(d,c.node,r,s,o,u,p),this.processChildNodes(c.node,t,s,r,i,m),y))):(r[d]=f,this.keepIntermediateTensors&&(this.clonedTensorsMap[d]=this.cloneTensorList(f)),this.checkTensorForDisposal(d,c.node,r,s,o,u,p),this.processChildNodes(c.node,t,s,r,i,m))}else this.processChildNodes(c.node,t,s,r,i,m)}return l}processChildNodes(e,t,s,r,i,o){e.children.forEach(u=>{const[p]=W(u.name,s);i[p]||!o.has(u.name)||(u.op==="Merge"?u.inputNames.some(m=>!!L(m,r,s))&&(i[p]=!0,t.push({contexts:s.currentContext,node:u})):u.inputNames.every(m=>!!L(m,r,s))&&(i[p]=!0,t.push({contexts:s.currentContext,node:u})))})}dispose(){Object.keys(this.weightMap).forEach(e=>this.weightMap[e].forEach(t=>t.dispose()))}checkInputShapeAndType(e){Object.keys(e).forEach(t=>{const s=e[t],[r]=F(t),i=this.graph.nodes[r];if(i.attrParams.shape&&i.attrParams.shape.value){const o=i.attrParams.shape.value,u=o.length===s.shape.length&&s.shape.every((p,m)=>o[m]===-1||o[m]===p);S(u,()=>`The shape of dict['${i.name}'] provided in model.execute(dict) must be [${o}], but was [${s.shape}]`)}i.attrParams.dtype&&i.attrParams.dtype.value&&S(s.dtype===i.attrParams.dtype.value,()=>`The dtype of dict['${i.name}'] provided in model.execute(dict) must be ${i.attrParams.dtype.value}, but was ${s.dtype}`)})}mapInputs(e){var t,s;const r={};for(const i in e){const o=(s=(t=this._signature)===null||t===void 0?void 0:t.inputs)===null||s===void 0?void 0:s[i];o!=null?r[o.name]=e[i]:r[i]=e[i]}return r}checkInputs(e){const t=Object.keys(e).filter(s=>{const[r]=F(s);return this.graph.nodes[r]==null});if(t.length>0)throw new Error(`The dict provided in model.execute(dict) has keys: [${t}] that are not part of graph`)}mapOutputs(e){return e.map(t=>{var s,r;const i=(r=(s=this._signature)===null||s===void 0?void 0:s.outputs)===null||r===void 0?void 0:r[t];return i!=null?i.name:t},{})}checkOutputs(e){e.forEach(t=>{const[s]=F(t);if(!this.graph.nodes[s])throw new Error(`The output '${t}' is not found in the graph`)})}}class Hu{constructor(e={},t={}){this.hashTableNameToHandle=e,this.hashTableMap=t}addHashTable(e,t){this.hashTableNameToHandle[e]=t.handle,this.hashTableMap[t.id]=t}getHashTableHandleByName(e){return this.hashTableNameToHandle[e]}getHashTableById(e){return this.hashTableMap[e]}dispose(){for(const e in this.hashTableMap)this.hashTableMap[e].clearAndClose(),delete this.hashTableMap[e];for(const e in this.hashTableNameToHandle)this.hashTableNameToHandle[e].dispose(),delete this.hashTableNameToHandle[e]}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 */const qu="?tfjs-format=file",Wu="model.json";class Uu{get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}constructor(e,t={},s=Wt){this.modelUrl=e,this.loadOptions=t,this.version="n/a",this.io=s,t==null&&(this.loadOptions={}),this.resourceManager=new Hu}findIOHandler(){const e=this.modelUrl;if(e.load!=null)this.handler=e;else if(this.loadOptions.requestInit!=null)this.handler=this.io.browserHTTPRequest(e,this.loadOptions);else{const t=this.io.getLoadHandlers(e,this.loadOptions);if(t.length===0)t.push(this.io.browserHTTPRequest(e,this.loadOptions));else if(t.length>1)throw new Error(`Found more than one (${t.length}) load handlers for URL '${[e]}'`);this.handler=t[0]}}load(){if(this.findIOHandler(),this.handler.load==null)throw new Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");const e=this.handler.load();return Te(e)?e.then(t=>this.loadSync(t)):this.loadSync(e)}loadSync(e){this.artifacts=e;const t=this.artifacts.modelTopology;let s=this.artifacts.signature;if(this.artifacts.userDefinedMetadata!=null){const i=this.artifacts.userDefinedMetadata;i.signature!=null&&(s=i.signature),i.structuredOutputKeys!=null&&(this.structuredOutputKeys=i.structuredOutputKeys)}this.signature=s,this.version=`${t.versions.producer}.${t.versions.minConsumer}`;const r=this.io.decodeWeights(this.artifacts.weightData,this.artifacts.weightSpecs);if(this.executor=new Se(ht.Instance.transformGraph(t,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(r),this.executor.resourceManager=this.resourceManager,e.modelInitializer!=null&&e.modelInitializer.node!=null){const i=ht.Instance.transformGraph(e.modelInitializer);this.initializer=new Se(i),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializerSignature=e.initializerSignature}return!0}async save(e,t){if(typeof e=="string"){const s=this.io.getSaveHandlers(e);if(s.length===0)throw new Error(`Cannot find any save handlers for URL '${e}'`);if(s.length>1)throw new Error(`Found more than one (${s.length}) save handlers for URL '${e}'`);e=s[0]}if(e.save==null)throw new Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return e.save(this.artifacts)}addStructuredOutputNames(e){if(this.structuredOutputKeys){const t=e instanceof be?[e]:e,s={};return t.forEach((r,i)=>s[this.structuredOutputKeys[i]]=r),s}return e}predict(e,t){const s=this.execute(e,this.outputNodes);return this.addStructuredOutputNames(s)}async predictAsync(e,t){const s=await this.executeAsync(e,this.outputNodes);return this.addStructuredOutputNames(s)}normalizeInputs(e){var t;if(!(e instanceof be)&&!Array.isArray(e)){const i=(t=this.signature)===null||t===void 0?void 0:t.inputs;if(i!=null)for(const o in i){const u=i[o];u.resourceId!=null&&(e[o]=this.resourceIdToCapturedInput[u.resourceId])}return e}e=Array.isArray(e)?e:[e];const s=Object.keys(this.resourceIdToCapturedInput).length;if(e.length+s!==this.inputNodes.length)throw new Error(`Input tensor count mismatch, the graph model has ${this.inputNodes.length-s} non-resource placeholders, while there are ${e.length} input tensors provided.`);let r=0;return this.inputNodes.reduce((i,o)=>{var u,p,m;const l=(m=(p=(u=this.signature)===null||u===void 0?void 0:u.inputs)===null||p===void 0?void 0:p[o])===null||m===void 0?void 0:m.resourceId;return l!=null?i[o]=this.resourceIdToCapturedInput[l]:i[o]=e[r++],i},{})}normalizeOutputs(e){return e=e||this.outputNodes,Array.isArray(e)?e:[e]}executeInitializerGraph(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.execute({},[]):this.initializer.execute({},Object.keys(this.initializerSignature.outputs))}async executeInitializerGraphAsync(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.executeAsync({},[]):this.initializer.executeAsync({},Object.keys(this.initializerSignature.outputs))}setResourceIdToCapturedInput(e){if(this.resourceIdToCapturedInput={},this.initializerSignature){const t=this.initializerSignature.outputs,s=Object.keys(t);for(let r=0;r<s.length;r++){const i=s[r],o=t[i];this.resourceIdToCapturedInput[o.resourceId]=e[r]}}}execute(e,t){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(this.executeInitializerGraph()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);const s=this.executor.execute(e,t);return s.length>1?s:s[0]}async executeAsync(e,t){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(await this.executeInitializerGraphAsync()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);const s=await this.executor.executeAsync(e,t);return s.length>1?s:s[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(e){return Object.keys(e).reduce((t,s)=>(t[s]=[e[s]],t),{})}dispose(){this.executor.dispose(),this.initializer&&(this.initializer.dispose(),this.resourceIdToCapturedInput&&os(this.resourceIdToCapturedInput)),this.resourceManager.dispose()}}async function Gu(a,e={},t=Wt){if(a==null)throw new Error("modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model");e==null&&(e={}),e.fromTFHub&&typeof a=="string"&&(a=Ku(a));const s=new Uu(a,e,t);return await s.load(),s}function Ku(a){return a.endsWith("/")||(a=a+"/"),`${a}${Wu}${qu}`}const Ju={class:"tensorflow"},Xu={class:"tensorflow_modelData_box"},Yu={class:"tensorflow_modelData_header"},Zu=["onUpdate:modelValue"],Qu=["onClick"],Mu={class:"tensorflow_modelData_body"},ep={class:"tensorflow_modelData_body__left"},tp={key:0,class:"tensorflow_modelData_body__left_box"},ap=["onClick"],sp={class:"tensorflow_modelData_body__right"},rp=["src"],np={class:"tensorflow_drill"},ip={key:1},op={key:2},up={class:"tensorflow_detection"},pp={style:{width:"100px",height:"20px"}},mp=Jt({__name:"TensorflowTwo",setup(a){const e=ke(-1),t=ke(),s=tt({width:260,height:260,addStatus:!1,MOBILE_NET_INPUT_HEIGHT:224,MOBILE_NET_INPUT_WIDTH:224,trainingDataInputs:[],trainingDataOutputs:[],isEnd:!1,isDrill:!1}),r=tt({name:[],data:[]});function i(){e.value=-1,m()}function o(){if(l.status){if(e.value===-1){const w=t.value;w.width=s.width,w.height=s.height;const C=w.getContext("2d");C.drawImage(p.value,0,0,s.width,s.height),R(function(){const O=lt(C.getImageData(0,0,s.width,s.height)).div(255),E=Le.resizeBilinear(O,[s.MOBILE_NET_INPUT_HEIGHT,s.MOBILE_NET_INPUT_WIDTH],!0);let V=h.predict(E.expandDims()),H=N.predict(V).squeeze(),ne=H.argMax().arraySync(),te=H.arraySync();console.log(H),console.log(ne),console.log(te),console.log("================="),r.data=te})}else{const w=t.value[0];w.width=s.width,w.height=s.height;const C=w.getContext("2d");if(C.drawImage(p.value,0,0,s.width,s.height),s.addStatus){d.value[e.value].data.push({base:w.toDataURL("image/png"),imageData:C.getImageData(0,0,s.width,s.height)});let O=!0;d.value.map(E=>{E.data.length||(O=!1)}),s.isDrill=O}}window.requestAnimationFrame(o)}}function u(w){s.addStatus=!0,document.onmouseup=function(){s.addStatus=!1,document.onmouseup=null}}const{newVideoRef:p,videoButtonClick:m,videoConfig:l}=ea({width:s.width,height:s.height,videoProceed:o}),c={name:"模型",data:[]},d=ke([]);function f(w){d.value=d.value.filter((C,O)=>O!==w)}function g(){const w=JSON.parse(JSON.stringify(c));w.name=w.name+(d.value.length+1),d.value.push(w)}for(let w=0;w<2;w++)g();function y(w){e.value=w,l.status||m()}let h;Xt(async()=>{h=await Gu("./tfjs-model_imagenet_mobilenet_v3_small_100_224_feature_vector_5_default_1",{fromTFHub:!0}),R(function(){h.predict(Vt([1,s.MOBILE_NET_INPUT_HEIGHT,s.MOBILE_NET_INPUT_WIDTH,3]))})});let N;const _=[],A=[];async function T(){us(_,A);let w=Pt(A,"int32"),C=Lt(w,d.value.length),O=re(_);await N.fit(O,C,{shuffle:!0,batchSize:5,epochs:10,callbacks:{onEpochEnd:(E,V)=>{console.log("Data for epoch "+E,V)}}}),w.dispose(),C.dispose(),O.dispose(),s.isDrill=!1,s.isEnd=!0,r.name=d.value.map(E=>E.name),y(-1)}function I(){for(let w=0;w<d.value.length;w++){const{name:C,data:O}=d.value[w];for(let E=0;E<O.length;E++){const V=R(function(){let H=lt(O[E].imageData),te=Le.resizeBilinear(H,[s.MOBILE_NET_INPUT_HEIGHT,s.MOBILE_NET_INPUT_WIDTH],!0).div(255);return h.predict(te.expandDims()).squeeze()});_.push(V),A.push(w)}}T()}function $(){s.isEnd=!1,N=bo(),N.add(ct({inputShape:[1024],units:128,activation:"relu"})),N.add(ct({units:d.value.length,activation:"softmax"})),N.summary(),N.compile({optimizer:"adam",loss:d.value.length===2?"binaryCrossentropy":"categoricalCrossentropy",metrics:["accuracy"]}),I()}return(w,C)=>(x(),j("div",Ju,[z("div",Xu,[(x(!0),j(Ae,null,De(d.value,(O,E)=>(x(),j("div",{key:E,class:"tensorflow_modelData"},[z("div",Yu,[Zt(z("input",{type:"text","onUpdate:modelValue":V=>O.name=V},null,8,Zu),[[Qt,O.name]]),z("span",{onClick:V=>f(E)},"x",8,Qu)]),z("div",Mu,[z("div",ep,[E===e.value?(x(),j("div",tp,[z("canvas",{ref_for:!0,ref_key:"canvasRef",ref:t},null,512),z("span",{onClick:i},"x"),z("div",{onMousedown:u}," 按住录制 ",32)])):(x(),j("button",{key:1,onClick:V=>y(E)},"开启摄像头",8,ap))]),z("div",sp,[z("h3",null,at(O.data.length),1),z("div",null,[(x(!0),j(Ae,null,De(O.data,(V,H)=>(x(),j("img",{key:H,src:V.base},null,8,rp))),128))])])])]))),128)),z("button",{onClick:g},"添加模型")]),z("div",np,[s.isDrill?(x(),j("button",{key:0,onClick:$},"训练模型")):s.isEnd?(x(),j("button",ip,"已训练模型")):(x(),j("button",op,"请添加数据"))]),z("div",up,[e.value===-1?(x(),j("canvas",{key:0,width:"260px",height:"260px",ref_key:"canvasRef",ref:t},null,512)):Yt("",!0),(x(!0),j(Ae,null,De(r.name,(O,E)=>(x(),j("div",{key:E},[z("span",null,at(O),1),z("div",pp,[z("div",{style:Mt({background:"red",width:(r.data[E]||0)*100+"%",height:"20px"})},null,4)])]))),128))])]))}});const yp=zn(mp,[["__scopeId","data-v-ac65ea80"]]);export{yp as default};
