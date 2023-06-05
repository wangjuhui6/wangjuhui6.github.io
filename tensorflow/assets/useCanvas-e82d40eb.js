import{x as y,Q as j,g as w,h3 as x,h4 as F,gU as ie,bV as R,h5 as oe,w as ae,G as le,h6 as ue,b1 as he,h7 as ce,h8 as fe,a0 as ge,h0 as de,h9 as pe,h1 as me,H as Se,N as Ie,g_ as Ee,$ as we,gZ as Ae,J as $e,D as be,gG as Me,ha as ye,hb as _e,E as ve,hc as Oe,a6 as Re,hd as xe,he as Ne,hf as Te,gt as De,hg as Ge,hh as Fe,gW as We,gM as N,gK as C,dF as z,hi as ke}from"./graph_model-9162a669.js";import{m as Le,f as v,h as Pe}from"./index-bce47476.js";function Z(e,t){const r=e.shape.length,n=t.shape.length;if(r<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${r}.`);if(n<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${n}.`);if(t.dtype!=="int32")throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[n-1]>r)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[n-1]} vs. ${r}`);if(y(e.shape)===0)throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${e.shape}.`);const s=t.shape,o=s[s.length-1];let i=1;for(let u=0;u<s.length-1;++u)i*=s[u];const a=e.shape,h=s.slice();h.pop();let c=1;for(let u=o;u<r;++u)c*=a[u],h.push(a[u]);const d=[...j(e.shape).map(u=>u/c),1].slice(0,o);return[h,i,c,d]}const Qt=Object.freeze(Object.defineProperty({__proto__:null,prepareAndValidate:Z},Symbol.toStringTag,{value:"Module"}));/**
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
 */const W=-2,Ve=-1;function Ce(e,t,r){const n=e.shape.length;w(n===t.length,()=>`Error in slice${n}D: Length of begin ${t} must match the rank of the array (${n}).`),w(n===r.length,()=>`Error in slice${n}D: Length of size ${r} must match the rank of the array (${n}).`);for(let s=0;s<n;++s)w(t[s]+r[s]<=e.shape[s],()=>`Error in slice${n}D: begin[${s}] + size[${s}] (${t[s]+r[s]}) would overflow input.shape[${s}] (${e.shape[s]})`)}function ze(e){const t=[];let r=0;for(;e>0;)e&1&&t.push(r),e/=2,r++;return t}function Ue(e,t,r){const n=[];for(let s=0;s<e.length;s++)n[s]=Math.ceil((t[s]-e[s])/r[s]);return n}function X(e,t,r,n){const s=[...e];for(let o=s.length;o<n.length;o++)s.push(1);for(let o=0;o<r;o++)o===0?s[t]=1:(s.splice(t,0,1),s.pop());return s}function q(e,t,r){return r<=e?r:r-(t-1)}function K(e,t){const r=[];for(let n=0;n<e;n++)r.push(t+n);return r}function Be(e,t,r,n,s,o,i,a,h){const c=e.length;let d=new Array(c),u=new Array(c),l=new Array(c);if(t.length&&r>0){const g=t[0],p=r+1;d=J(i,g,p,n,e),u=Q(a,g,p,s,e),l=X(o,g,p,e)}else for(let g=0;g<c;g++)d[g]=ee(i,n,o,e,g,h),u[g]=te(a,s,o,e,g,h),l[g]=Y(o,g,h);return{begin:d,end:u,strides:l}}function J(e,t,r,n,s){const o=[...s],i=K(r,t);for(let a=0;a<o.length;a++)if(i.indexOf(a)>-1)o[a]=0;else{const h=q(t,r,a);let c=n[h];e&1<<h&&(c=0),o[a]=c}return o}function Q(e,t,r,n,s){const o=[...s],i=K(r,t);for(let a=0;a<o.length;a++)if(i.indexOf(a)>-1)o[a]=Number.MAX_SAFE_INTEGER;else{const h=q(t,r,a);let c=n[h];e&1<<h&&(c=Number.MAX_SAFE_INTEGER),o[a]=c}for(let a=0;a<o.length;a++){const h=s[a];o[a]<0&&(o[a]+=h),o[a]=x(0,o[a],s[a])}return o}function Y(e,t,r){let n=e[t];return(r&1<<t||n==null)&&(n=1),n}function ee(e,t,r,n,s,o){let i=t[s];const a=r[s]||1;(e&1<<s||o&1<<s||i==null)&&(a>0?i=Number.MIN_SAFE_INTEGER:i=Number.MAX_SAFE_INTEGER);const h=n[s];return i<0&&(i+=h),i=x(0,i,h-1),i}function te(e,t,r,n,s,o){let i=t[s];const a=r[s]||1;(e&1<<s||o&1<<s||i==null)&&(a>0?i=Number.MAX_SAFE_INTEGER:i=Number.MIN_SAFE_INTEGER);const h=n[s];return i<0&&(i+=h),a>0?i=x(0,i,h):i=x(-1,i,h-1),i}function ne(e,t,r){let n=r.length;for(let s=0;s<r.length;s++)if(r[s]>1){n=s;break}for(let s=n+1;s<r.length;s++)if(t[s]>0||r[s]!==e[s])return!1;return!0}function re(e,t){let r=e.length>0?e[e.length-1]:1;for(let n=0;n<e.length-1;n++)r+=e[n]*t[n];return r}function He(e,t,r){let n;const s=e.shape.length;typeof t=="number"?n=[t,...new Array(s-1).fill(0)]:t.length<s?n=t.concat(new Array(s-t.length).fill(0)):n=t.slice(),n.forEach(i=>{w(i!==-1,()=>"slice() does not support negative begin indexing.")});let o;return r==null?o=new Array(s).fill(-1):typeof r=="number"?o=[r,...new Array(s-1).fill(-1)]:r.length<s?o=r.concat(new Array(s-r.length).fill(-1)):o=r,o=o.map((i,a)=>i>=0?i:(w(i===-1,()=>`Negative size values should be exactly -1 but got ${i} for the slice() size at index ${a}.`),e.shape[a]-n[a])),[n,o]}function je(e,t,r,n,s,o,i,a,h){let c;if(n==null?(c=new Array(t.length),c.fill(1)):c=n,i!=null&&i&i-1)throw new Error("Multiple ellipses in slice is not allowed.");let d=!1;const u={dims:c.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:r.slice(),strides:c.slice(),beginMask:s,endMask:o,ellipsisMask:i,newAxisMask:a,shrinkAxisMask:h};for(let f=0;f<u.dims;f++)d&&1<<f&a&&u.numAddAxisAfterEllipsis++,1<<f&i&&(d=!0);d||(u.ellipsisMask|=1<<u.dims,u.dims++);const l={dims:e.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};Ze(u,l);let g=!0,p=!0,I=!0;const E=[],m=[];for(let f=0;f<e.length;++f){if(l.strides[f]===0)throw Error(`strides[${f}] must be non-zero`);const $=!!(l.shrinkAxisMask&1<<f),A=e[f];if(A===-1){E.push($?1:-1);continue}const P=[l.beginMask&1<<f,l.endMask&1<<f],V=[l.strides[f]>0?0:-1,l.strides[f]>0?A:A-1];if($&&l.strides[f]<=0)throw Error("only stride 1 allowed on non-range indexing.");I=I&&l.strides[f]===1;const T=!!(l.beginMask&1<<f&&l.endMask&1<<f);if(l.beginValid&&l.endValid){if($){const D=l.begin[f]<0?A+l.begin[f]:l.begin[f];if(l.begin[f]=D,l.end[f]=l.begin[f]+1,D<0||D>=A)throw Error(`slice index ${l.begin[f]} of dimension ${f} out of bounds.`)}else l.begin[f]=U(l.begin[f],0,l.strides[f],A,P,V),l.end[f]=U(l.end[f],1,l.strides[f],A,P,V);const _=l.strides[f]===1&&l.begin[f]===0&&l.end[f]===A;g=g&&_,p=p&&(f===0&&l.strides[f]===1||_)}else g=g&&l.strides[f]===1&&T,p=p&&(f===0&&l.strides[f]===1||T);let M,O=!1;if(l.beginValid&&l.endValid?(M=l.end[f]-l.begin[f],O=!0):$?(M=1,O=!0):T&&A>=0&&(l.strides[f]<0?M=-A:M=A,O=!0),O){let _;M===0||M<0!=l.strides[f]<0?_=0:_=Math.trunc(M/l.strides[f])+(M%l.strides[f]!==0?1:0),E.push(_)}else E.push(-1)}for(let f=0;f<l.finalShapeGatherIndices.length;++f){const $=l.finalShapeGatherIndices[f];$>=0?m.push(E[$]):$===W&&m.push(1)}return{finalShapeSparse:m.filter((f,$)=>l.finalShapeGatherIndices[$]!==W),finalShape:m,isIdentity:g,sliceDim0:p,isSimpleSlice:I,begin:l.begin,end:l.end,strides:l.strides}}function Ze(e,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let r=0;t.beginValid=e.begin!=null,t.endValid=e.end!=null,t.begin=new Array(t.dims),t.end=new Array(t.dims),t.strides=new Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=new Array(t.dims);for(let n=0;n<e.dims;n++)if(1<<n&e.ellipsisMask){const s=Math.min(t.dims-(e.dims-n)+1+e.numAddAxisAfterEllipsis,t.dims);for(;r<s;r++)t.begin[r]=0,t.end[r]=0,t.strides[r]=1,t.beginMask|=1<<r,t.endMask|=1<<r,t.finalShapeGatherIndices.push(r),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[r]=n}else if(1<<n&e.newAxisMask)t.finalShapeGatherIndices.push(W),t.finalShapeGatherIndicesSparse.push(-1);else{if(r===t.begin.length)throw Error(`Index out of range using input dim ${r}; input has only ${t.dims} dims, ${t.begin.length}.`);e.begin!=null&&(t.begin[r]=e.begin[n]),e.end!=null&&(t.end[r]=e.end[n]),t.strides[r]=e.strides[n],e.beginMask&1<<n&&(t.beginMask|=1<<r),e.endMask&1<<n&&(t.endMask|=1<<r),e.shrinkAxisMask&1<<n?(t.finalShapeGatherIndices.push(Ve),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<r):(t.finalShapeGatherIndices.push(r),t.finalShapeGatherIndicesSparse.push(n)),t.inputShapeGatherIndicesSparse[r]=n,r++}}function U(e,t,r,n,s,o){if(s[t])return r>0?o[t]:o[t+1&1];{const i=e<0?n+e:e;return i<o[0]?o[0]:i>o[1]?o[1]:i}}const Xe=Object.freeze(Object.defineProperty({__proto__:null,assertParamsValid:Ce,computeFlatOffset:re,computeOutShape:Ue,getNormalizedAxes:Be,isSliceContinous:ne,maskToAxes:ze,parseSliceParams:He,sliceInfo:je,startForAxis:ee,startIndicesWithElidedDims:J,stopForAxis:te,stopIndicesWithElidedDims:Q,stridesForAxis:Y,stridesWithElidedDims:X},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qe(e,t){const r=e[0].length;e.forEach((s,o)=>{w(s.length===r,()=>`Error in concat${r}D: rank of tensors[${o}] must be the same as the rank of the rest (${r})`)}),w(t>=0&&t<r,()=>`Error in concat${r}D: axis must be between 0 and ${r-1}.`);const n=e[0];e.forEach((s,o)=>{for(let i=0;i<r;i++)w(i===t||s[i]===n[i],()=>`Error in concat${r}D: Shape of tensors[${o}] (${s}) does not match the shape of the rest (${n}) along the non-concatenated axis ${o}.`)})}function Ke(e,t){const r=e[0].slice();for(let n=1;n<e.length;n++)r[t]+=e[n][t];return r}/**
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
 */var b;(function(e){e[e.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",e[e.VALUE_ROWIDS=1]="VALUE_ROWIDS",e[e.ROW_LENGTHS=2]="ROW_LENGTHS",e[e.ROW_SPLITS=3]="ROW_SPLITS",e[e.ROW_LIMITS=4]="ROW_LIMITS",e[e.ROW_STARTS=5]="ROW_STARTS"})(b||(b={}));function Je(e,t,r){let n=new Array;if(r==null&&t==null)return n;if(t==null)for(;n.length<e+r.length;)n.push(-1);else n=t.slice();if(r==null)return n;if(e+r.length!==n.length)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${e+r.length}, but shape.rank = ${n.length}`);for(let s=1;s<r.length;++s){const o=r[s],i=n[n.length-r.length+s],a=n[i];if(o>=0)if(a>=0){if(a!==o)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${s+e}] = ${o} but shape[${s+e}] = ${a}`)}else n[i]=o}return n}function Qe(e){const t={FIRST_DIM_SIZE:b.FIRST_DIM_SIZE,VALUE_ROWIDS:b.VALUE_ROWIDS,ROW_LENGTHS:b.ROW_LENGTHS,ROW_SPLITS:b.ROW_SPLITS,ROW_LIMITS:b.ROW_LIMITS,ROW_STARTS:b.ROW_STARTS},r=[];for(const n of e)if(n in t)r.push(t[n]);else break;return r}function Ye(e){return e.length===0?0:e[0]===b.FIRST_DIM_SIZE?e.length-1:e.length}function et(e,t){if(e==null||t==null)return;const r=e.length,n=t.length;if(r>=n)throw new Error(`defaultValue.shape=${e} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${r} must be less than ragged tensor input flatValues.rank = ${n})`);for(let s=0;s<Math.min(r,n-1);++s){const o=e[s],i=t[s+1];if(o>=0&&i>=0&&o!==1&&o!==i)throw new Error(`defaultValue.shape=${e}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${s-e.length}] = ${o} but ragged tensor input.flatValues.shape[${s-e.length}] = ${i}`)}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const k=30;function tt(e){return e<=k?e:F(e,Math.floor(Math.sqrt(e)))}/**
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
 */function nt(e,t,r){const n=r*(typeof e=="number"?e:e[0]),s=t*(typeof e=="number"?e:e[1]);return[n,s]}/**
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
 */function rt(e,t,r,n=!0){let s=[];if(n)s=s.concat(t.slice(0)),s.push(e[0]/r),s=s.concat(e.slice(1));else{s=s.concat(e[0]);const o=t.length;for(let i=0;i<o;++i)s=s.concat([e[i+1]/t[i],t[i]]);s=s.concat(e.slice(o+1))}return s}function st(e,t,r=!0){const n=[];if(r){n.push(t);for(let s=t+1;s<e;++s)s<=2*t?(n.push(s),n.push(s-(t+1))):n.push(s)}else{const s=[],o=[];for(let i=1;i<e;++i)i>=t*2+1||i%2===1?o.push(i):s.push(i);n.push(...s),n.push(0),n.push(...o)}return n}function it(e,t,r,n=!0){const s=[];n?s.push(e[0]/r):s.push(e[0]*r);for(let o=1;o<e.length;++o)o<=t.length?n?s.push(t[o-1]*e[o]):s.push(e[o]/t[o-1]):s.push(e[o]);return s}function ot(e,t){const r=[0];for(let n=0;n<t;++n)r.push(e[n][0]);return r}function at(e,t,r){const n=e.slice(0,1);for(let s=0;s<r;++s)n.push(e[s+1]-t[s][0]-t[s][1]);return n}/**
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
 */const lt=1.7580993408473768,ut=1.0507009873554805;/**
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
 */const ht=.3275911,ct=.254829592,ft=-.284496736,gt=1.421413741,dt=-1.453152027,pt=1.061405429;/**
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
 */function mt(e,t){if(e.length!==t.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${e.length}, imag: ${t.length}.`);const r=new Float32Array(e.length*2);for(let n=0;n<r.length;n+=2)r[n]=e[n/2],r[n+1]=t[n/2];return r}function St(e){const t=new Float32Array(e.length/2),r=new Float32Array(e.length/2);for(let n=0;n<e.length;n+=2)t[n/2]=e[n],r[n/2]=e[n+1];return{real:t,imag:r}}function It(e){const t=Math.ceil(e.length/4),r=new Float32Array(t),n=new Float32Array(t);for(let s=0;s<e.length;s+=4)r[Math.floor(s/4)]=e[s],n[Math.floor(s/4)]=e[s+1];return{real:r,imag:n}}function Et(e){const t=Math.floor(e.length/4),r=new Float32Array(t),n=new Float32Array(t);for(let s=2;s<e.length;s+=4)r[Math.floor(s/4)]=e[s],n[Math.floor(s/4)]=e[s+1];return{real:r,imag:n}}function wt(e,t){const r=e[t*2],n=e[t*2+1];return{real:r,imag:n}}function At(e,t,r,n){e[n*2]=t,e[n*2+1]=r}function $t(e,t){const r=new Float32Array(e/2),n=new Float32Array(e/2);for(let s=0;s<Math.ceil(e/2);s++){const o=(t?2:-2)*Math.PI*(s/e);r[s]=Math.cos(o),n[s]=Math.sin(o)}return{real:r,imag:n}}function bt(e,t,r){const n=(r?2:-2)*Math.PI*(e/t),s=Math.cos(n),o=Math.sin(n);return{real:s,imag:o}}/**
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
 */const G="->",Mt=/->/g,B=",",H="...";function yt(e,t){e=e.replace(/\s/g,"");const r=(e.length-e.replace(Mt,"").length)/G.length;if(r<1)throw new Error("Equations without an arrow are not supported.");if(r>1)throw new Error(`Equation must contain exactly one arrow ("${G}").`);const[n,s]=e.split(G);w(n.indexOf(H)===-1,()=>`The ellipsis notation ("${H}") is not supported yet.`);const o=n.split(B),i=o.length;if(t!==i)throw new Error(`Expected ${i} input tensors, received ${t}`);if(i>2)throw new Error("Support for more than 2 input tensors is not implemented yet.");const a=[];for(let l=0;l<s.length;++l){const g=s[l];if(!o.some(p=>p.indexOf(g)!==-1))throw new Error(`Output subscripts contain the label ${g} not present in the input subscripts.`);a.indexOf(g)===-1&&a.push(g)}for(let l=0;l<n.length;++l){const g=n[l];a.indexOf(g)===-1&&g!==B&&a.push(g)}const h=new Array(o.length);for(let l=0;l<i;++l){if(new Set(o[l].split("")).size!==o[l].length)throw new Error(`Found duplicate axes in input component ${o[l]}. Support for duplicate axes in input is not implemented yet.`);h[l]=[];for(let g=0;g<o[l].length;++g)h[l].push(a.indexOf(o[l][g]))}const c=a.length,d=s.length,u=[];for(let l=d;l<c;++l)u.push(l);return{allDims:a,summedDims:u,idDims:h}}function _t(e,t){let r=new Array(e);r.fill(-1);for(let s=0;s<t.length;++s)r[t[s]]=s;const n=[];for(let s=0;s<e;++s)r[s]===-1&&n.push(s);return r=r.filter(s=>s!==-1),{permutationIndices:r,expandDims:n}}function vt(e,t,r){const n=new Array(e);for(let s=0;s<r.length;++s){const o=r[s].shape;for(let i=0;i<t[s].length;++i)n[t[s][i]]===void 0?n[t[s][i]]=o[i]:w(n[t[s][i]]===o[i],()=>`Expected dimension ${n[t[s][i]]} at axis ${i} of input shaped ${JSON.stringify(o)}, but got dimension ${o[i]}`)}}function Ot(e,t){const r=e,n=[];let s=0;e.length===0&&r.push(-1),s=e.length+1;for(let i=0;i<s;++i)n.push([]);const o=[];for(let i=0;i<r.length;++i){const a=r[i],h=xt(t,a);for(const c of h)o.indexOf(c)===-1&&(n[i].push(c),o.push(c))}return{path:r,steps:n}}function Rt(e){return e.every((t,r)=>t===r)}function xt(e,t){const r=[];for(let n=0;n<e.length;++n)(e[n].length===0||e[n].indexOf(t)!==-1||t===-1)&&r.push(n);return r}function Nt(e,t,r=0){let n=[];if(typeof t=="number")w(e.shape[r]%t===0,()=>"Number of splits must evenly divide the axis."),n=new Array(t).fill(e.shape[r]/t);else{const s=t.reduce((i,a)=>(a===-1&&(i+=1),i),0);w(s<=1,()=>"There should be only one negative value in split array.");const o=t.indexOf(-1);if(o!==-1){const i=t.reduce((a,h)=>h>0?a+h:a);t[o]=e.shape[r]-i}w(e.shape[r]===t.reduce((i,a)=>i+a),()=>"The sum of sizes must match the size of the axis dimension."),n=t}return n}/**
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
 */function Tt(e){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${e}`}function Dt(e,t){return`indices(${e}, 0) is invalid: ${t} < 0`}function Gt(e,t,r){return`indices(${e}, 0) is invalid: ${t} >= ${r}`}/**
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
 */function Ft(e,t){return`only one output dimension may be -1, not both ${e} and ${t}`}function Wt(e,t){return`size ${e} must be non-negative, not ${t}`}function kt(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function Lt(e,t){const r=y(e),n=y(t);return`Input to reshape is a SparseTensor with ${r}
  dense values, but the requested shape requires a multiple of ${n}. inputShape=${e} outputShape= ${t}`}function Pt(e,t){const r=y(e),n=y(t);return`Input to reshape is a tensor with ${r} dense values, but the requested shape has ${n}. inputShape=${e} outputShape=${t}`}/**
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
 */function Vt(){return"segment ids must be >= 0"}function Ct(){return"segment ids are not increasing"}function zt(e,t){return`Segment id ${e} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function Ut(e,t,r){return`Bad: indices[${e}] == ${t} out of range [0, ${r})`}/**
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
 */function Bt(e,t){let r=!1,n;for(e<=k?(n=e,r=!0):n=F(e,Math.floor(Math.sqrt(e)));!r;)n>t||n===e?r=!0:n=F(e,n+1);return n}function Ht(e,t,r){const n=[],s=e.length;for(let o=0;o<s;o++)o!==t?n.push(e[o]):n.push(r);return n}function jt(e,t,r,n){const s=t.shape.length,o=e.shape.length;if(n!==0&&(n<-s||n>s))throw new Error(`Expect batchDims in the range of [-${s}, ${s}], but got ${n}`);if(n<0&&(n+=s),n>o)throw new Error(`batchDims (${n}) must be less than rank(x) (
    ${o}).`);if(r<n)throw new Error(`batchDims (${n}) must be less than or equal to axis (${r}).`);for(let u=0;u<n;++u)if(e.shape[u]!==t.shape[u])throw new Error(`x.shape[${u}]: ${e.shape[u]} should be equal to indices.shape[${u}]: ${t.shape[u]}.`);const i=e.shape[r],a=[];let h=1,c=1,d=1;for(let u=0;u<n;++u)a.push(e.shape[u]),h*=e.shape[u];for(let u=n;u<r;u++)a.push(e.shape[u]),c*=e.shape[u];for(let u=n;u<s;u++)a.push(t.shape[u]);for(let u=r+1;u<o;u++)a.push(e.shape[u]),d*=e.shape[u];return{batchSize:h,sliceSize:d,outerSize:c,dimSize:i,outputShape:a}}const Zt=Object.freeze(Object.defineProperty({__proto__:null,collectGatherOpShapeInfo:jt,computeOutShape:Ht,segOpComputeOptimalWindowSize:Bt},Symbol.toStringTag,{value:"Module"}));/**
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
 */function L(e){try{return e.map(t=>ie(t))}catch(t){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${t}`)}}function se(e){return e.map(t=>R(t))}const Yt=Object.freeze(Object.defineProperty({__proto__:null,ERF_A1:ct,ERF_A2:ft,ERF_A3:gt,ERF_A4:dt,ERF_A5:pt,ERF_P:ht,PARALLELIZE_THRESHOLD:k,get RowPartitionType(){return b},SELU_SCALE:ut,SELU_SCALEALPHA:lt,applyActivation:oe,assertAndGetBroadcastShape:ae,assertAxesAreInnerMostDims:le,assertParamsConsistent:qe,assignToTypedArray:At,axesAreInnerMostDims:ue,calculateShapes:he,checkEinsumDimSizes:vt,checkPadOnDimRoundingMode:ce,combineLocations:fe,combineRaggedTensorToTensorShapes:Je,complexWithEvenIndex:It,complexWithOddIndex:Et,computeConv2DInfo:ge,computeConv3DInfo:de,computeDefaultPad:pe,computeDilation2DInfo:me,computeOptimalWindowSize:tt,computeOutAndReduceShapes:Se,computeOutShape:Ke,computePool2DInfo:Ie,computePool3DInfo:Ee,convertConv2DDataFormat:we,decodeEinsumEquation:yt,eitherStridesOrDilationsAreOne:Ae,expandShapeToKeepDim:$e,exponent:bt,exponents:$t,fromStringArrayToUint8:se,fromUint8ToStringArray:L,getAxesPermutation:be,getBroadcastDims:Me,getComplexWithIndex:wt,getEinsumComputePath:Ot,getEinsumPermutation:_t,getFusedBiasGradient:ye,getFusedDyActivation:_e,getImageCenter:nt,getInnerMostAxes:ve,getPermuted:st,getRaggedRank:Ye,getReductionAxes:Oe,getReshaped:rt,getReshapedPermuted:it,getRowPartitionTypesHelper:Qe,getSliceBeginCoords:ot,getSliceSize:at,getSparseFillEmptyRowsIndicesDenseShapeMismatch:Tt,getSparseFillEmptyRowsNegativeIndexErrorMessage:Dt,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:Gt,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:kt,getSparseReshapeInputOutputMismatchErrorMessage:Pt,getSparseReshapeInputOutputMultipleErrorMessage:Lt,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:Ft,getSparseReshapeNegativeOutputDimErrorMessage:Wt,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:Ut,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:Vt,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:Ct,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:zt,getUndoAxesPermutation:Re,isIdentityPermutation:Rt,log:xe,mergeRealAndImagArrays:mt,prepareAndValidate:Z,prepareSplitSize:Nt,segment_util:Zt,shouldFuse:Ne,slice_util:Xe,splitRealAndImagArrays:St,tupleValuesAreOne:Te,upcastType:De,validateDefaultValueShape:et,validateInput:Ge,validateUpdateShape:Fe,warn:We},Symbol.toStringTag,{value:"Module"}));/**
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
 */function en(e,t,r,n){const s=N(r,y(t));if(n&&r!=="string"){let o=0;e.forEach(i=>{const a=y(i.shape);s.set(i.vals,o),o+=a})}else{let o=0;e.forEach(i=>{const a=r==="string"?L(i.vals):i.vals;let h=0;for(let c=0;c<i.shape[0];++c){const d=c*t[1]+o;for(let u=0;u<i.shape[1];++u)s[d+u]=a[h++]}o+=i.shape[1]})}return s}/**
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
 */function tn(e,t,r,n){const s=e===t,o=e<t&&r<0,i=t<e&&r>1;if(s||o||i)return C(0,n);const a=Math.abs(Math.ceil((t-e)/r)),h=C(a,n);t<e&&r===1&&(r=-1),h[0]=e;for(let c=1;c<h.length;c++)h[c]=h[c-1]+r;return h}/**
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
 */function nn(e,t,r,n,s){const o=ne(n,t,r),i=y(r),a=j(n);if(o){const u=re(t,a);return s==="string"?e.slice(u,u+i):e.subarray(u,u+i)}const h=s==="string"?L(e):e,c=z(n,s,h),d=z(r,s);for(let u=0;u<d.size;++u){const l=d.indexToLoc(u),g=l.map((p,I)=>p+t[I]);d.set(c.get(...g),...l)}return s==="string"?se(d.values):d.values}/**
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
 */class Xt{constructor(t,r,n,s,o,i){this.separator=R(t),this.nGramWidths=r,this.leftPad=R(n),this.rightPad=R(s),this.padWidth=o,this.preserveShort=i}getPadWidth(t){return Math.min(this.padWidth<0?t-1:this.padWidth,t-1)}getNumNGrams(t,r){const n=this.getPadWidth(r);return Math.max(0,t+2*n-r+1)}createNGrams(t,r,n,s,o,i){for(let a=0;a<o;++a){const h=this.getPadWidth(i),c=Math.max(0,h-a),d=Math.max(0,h-(o-(a+1))),u=i-(c+d),l=r+(c>0?0:a-h);let g=0;g+=c*this.leftPad.length;for(let S=0;S<u;++S)g+=t[l+S].length;g+=d*this.rightPad.length;const p=c+d+u-1;g+=p*this.separator.length,n[s+a]=new Uint8Array(g);const I=n[s+a];let E=0;const m=S=>S.forEach(f=>I[E++]=f);for(let S=0;S<c;++S)m(this.leftPad),m(this.separator);for(let S=0;S<u-1;++S)m(t[l+S]),m(this.separator);if(u>0){m(t[l+u-1]);for(let S=0;S<d;++S)m(this.separator),m(this.rightPad)}else{for(let S=0;S<d-1;++S)m(this.rightPad),m(this.separator);m(this.rightPad)}}}compute(t,r){const n=t.length,s=r.length;if(s>0){let h=r[0];if(h!==0)throw new Error(`First split value must be 0, got ${h}`);for(let c=1;c<s;++c){let d=r[c]>=h;if(d=d&&r[c]<=n,!d)throw new Error(`Invalid split value ${r[c]}, must be in [${h}, ${n}]`);h=r[c]}if(h!==n)throw new Error(`Last split value must be data size. Expected ${n}, got ${h}`)}const o=s-1,i=N("int32",s);if(n===0||s===0){const h=new Array(n);for(let c=0;c<=o;++c)i[c]=0;return[h,i]}i[0]=0;for(let h=1;h<=o;++h){const c=r[h]-r[h-1];let d=0;this.nGramWidths.forEach(u=>{d+=this.getNumNGrams(c,u)}),this.preserveShort&&c>0&&d===0&&(d=1),i[h]=i[h-1]+d}const a=new Array(i[o]);for(let h=0;h<o;++h){const c=r[h];let d=i[h];if(this.nGramWidths.forEach(u=>{const l=r[h+1]-r[h],g=this.getNumNGrams(l,u);this.createNGrams(t,c,a,d,g,u),d+=g}),this.preserveShort&&d===i[h]){const u=r[h+1]-r[h];if(u===0)continue;const l=u+2*this.padWidth,g=1;this.createNGrams(t,c,a,d,g,l)}}return[a,i]}}function rn(e,t,r,n,s,o,i,a){return new Xt(r,n,s,o,i,a).compute(e,t)}/**
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
 */function qt(e,t,r,n){if(!e.length)return;if(t.length===0){for(let o=0;o<e.length;++o)n.push(e.subarray(o,o+1));return}if(t.length===1){const o=t[0];let i=e.indexOf(o);for(;i!==-1;){const a=e.subarray(0,i);(!r||a.length!==0)&&n.push(a),e=e.subarray(i+1),i=e.indexOf(o)}(!r||e.length!==0)&&n.push(e);return}let s=0;for(let o=0;o<e.length+1;o++)if(o===e.length||t.indexOf(e[o])!==-1){const i=e.subarray(s,o);(!r||i.length!==0)&&n.push(i),s=o+1}}function sn(e,t,r){const n=e.length,s=[];let o=0,i=0;const a=new Array(n);for(let l=0;l<n;++l){const g=s.length;qt(e[l],t,r,s);const p=s.length-g;a[l]=p,o+=p,i=Math.max(i,p)}const h=N("int32",o*2),c=new Array(o),d=[n,i];let u=0;for(let l=0;l<n;++l)for(let g=0;g<a[l];++g)h[u*2]=l,h[u*2+1]=g,c[u]=s[u],++u;return[h,c,d]}/**
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
 */function on(e,t){const r=N("int32",e.length);for(let n=0;n<e.length;++n)r[n]=ke(e[n]).modulo(t).getLowBitsUnsigned();return r}function an({canvasRef:e,width:t=300,height:r=300}){const n=e?Le(e)?e:v(e):v(document.createElement("canvas")),s=v(),o=v(),i=v();Pe(()=>{e||(n.value.width=t,n.value.height=r);const{width:p,height:I}=n.value;s.value=p,o.value=I,i.value=n.value.getContext("2d")});function a(p=0,I=0,E=s.value,m=o.value){return i.value.getImageData(p,I,E,m)}function h(p="image/png"){return n.value.toDataURL(p)}function c(p,I=0,E=0,m=s.value,S=o.value){i.value.drawImage(p,I,E,m,S)}function d(p,I,E,m){m&&(i.value.fillStyle=m),i.value.beginPath(),i.value.arc(p,I,E,0,2*Math.PI),i.value.fill()}function u(p,I){I&&(i.value.fillStyle=I),i.value.beginPath(),i.value.rect(p[0],p[1],p[2],p[3]),i.value.fill()}function l({font:p,text:I,style:E,bbox:m,textBaseline:S="top"}){p&&(i.value.font=p),E&&(i.value.strokeStyle=E),i.value.textBaseline=S,i.value.strokeText(I,m[0],m[1])}function g({lineWidth:p=1,color:I,data:E}){i.value.beginPath(),I&&(i.value.strokeStyle=I),p&&(i.value.lineWidth=p),E.forEach((m,S)=>{S?i.value.lineTo(m[0],m[1]):i.value.moveTo(m[0],m[1])}),i.value.stroke()}return{newCanvasRef:n,ctx:i,CanvasWidth:s,CanvasHeight:o,getImageData:a,canvasToImg:h,drawImage:c,fillArc:d,drawLine:g,fillRect:u,drawText:l}}export{ht as $,Ut as A,zt as B,Ct as C,Nt as D,je as E,Ue as F,rn as G,sn as H,on as I,an as J,Yt as K,Qt as L,Xe as M,Qe as N,Ye as O,et as P,Je as Q,b as R,mt as S,L as T,tt as U,Ce as V,yt as W,vt as X,Ot as Y,_t as Z,Rt as _,st as a,ct as a0,ft as a1,gt as a2,dt as a3,pt as a4,lt as a5,ut as a6,Ht as a7,Bt as a8,it as b,re as c,ot as d,at as e,qe as f,rt as g,Ke as h,ne as i,en as j,se as k,Z as l,jt as m,nt as n,Gt as o,He as p,Dt as q,tn as r,nn as s,Tt as t,Pt as u,Lt as v,Wt as w,Ft as x,kt as y,Vt as z};
