import{fs as Ra,ft as za,c5 as ns}from"./index-c5cb974d.js";/**
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
 */const Va=1e-7,ja=1e-4;class Fw{constructor(t,n){this.backend=t,this.dataMover=n,this.data=new WeakMap,this.dataIdsCount=0}get(t){return this.data.has(t)||this.dataMover.moveData(this.backend,t),this.data.get(t)}set(t,n){this.dataIdsCount++,this.data.set(t,n)}has(t){return this.data.has(t)}delete(t){return this.dataIdsCount--,this.data.delete(t)}numDataIds(){return this.dataIdsCount}}class Ka{refCount(t){return lt("refCount")}incRef(t){return lt("incRef")}timerAvailable(){return!0}time(t){return lt("time")}read(t){return lt("read")}readSync(t){return lt("readSync")}readToGPU(t,n){return lt("readToGPU")}numDataIds(){return lt("numDataIds")}disposeData(t,n){return lt("disposeData")}write(t,n,s){return lt("write")}move(t,n,s,r,a){return lt("move")}memory(){return lt("memory")}floatPrecision(){return lt("floatPrecision")}epsilon(){return this.floatPrecision()===32?Va:ja}dispose(){return lt("dispose")}}function lt(e){throw new Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
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
 */function rr(e){let t=e.length,n=0;for(;t>0;)n=Math.random()*t|0,t--,We(e,t,n)}function Ua(e,t){if(e.length!==t.length)throw new Error(`Array sizes must match to be shuffled together First array length was ${e.length}Second array length was ${t.length}`);let n=e.length,s=0;for(;n>0;)s=Math.random()*n|0,n--,We(e,n,s),We(t,n,s)}function Wa(e,t,n){return Math.max(e,Math.min(t,n))}function qa(e){return e%2===0?e:e+1}function We(e,t,n){const s=e[t];e[t]=e[n],e[n]=s}function Ha(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function Ga(e,t){const n=Math.random();return t*n+(1-n)*e}function Ma(e,t){let n=0;for(let s=0;s<e.length;s++){const r=Number(e[s])-Number(t[s]);n+=r*r}return n}function d(e,t){if(!e)throw new Error(typeof t=="string"?t:t())}function at(e,t,n=""){d(Et(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function ee(e){d(e!=null,()=>"The input to the tensor constructor must be a non-null value.")}function Oe(e,t=[],n=!1){if(t==null&&(t=[]),Array.isArray(e)||yt(e)&&!n)for(let s=0;s<e.length;++s)Oe(e[s],t,n);else t.push(e);return t}function q(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function Ya(e){return e.length===0}function Et(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function he(e){return e%1===0}function Xa(e){if(Math.tanh!=null)return Math.tanh(e);if(e===1/0)return 1;if(e===-1/0)return-1;{const t=Math.exp(2*e);return(t-1)/(t+1)}}function Ja(e){const t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function Za(e){const t=new Uint32Array(e);for(let n=0;n<e;++n)t[n]=n;return rr(t),t}function Se(e,t){return t<=e.length?e:e+" ".repeat(t-e.length)}function Qa(e,t=r=>0,n,s=setTimeout){return new Promise((r,a)=>{let o=0;const i=()=>{if(e()){r();return}o++;const u=t(o);if(n!=null&&o>=n){a();return}s(i,u)};i()})}function to(e,t){let n=1,s=-1;for(let a=0;a<e.length;++a)if(e[a]>=0)n*=e[a];else if(e[a]===-1){if(s!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${s} and dim ${a}`);s=a}else if(e[a]<0)throw Error(`Shapes can not be < 0. Found ${e[a]} at dim ${a}`);if(s===-1){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(n===0)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);const r=e.slice();return r[s]=t/n,r}function Ce(e,t){const n=t.length;return e=e==null?t.map((s,r)=>r):[].concat(e),d(e.every(s=>s>=-n&&s<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),d(e.every(s=>he(s)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(s=>s<0?n+s:s)}function ar(e,t){const n=[],s=[],r=t!=null&&Array.isArray(t)&&t.length===0,a=t==null||r?null:Ce(t,e).sort();let o=0;for(let i=0;i<e.length;++i){if(a!=null){if(a[o]===i&&e[i]!==1)throw new Error(`Can't squeeze axis ${i} since its dim '${e[i]}' is not 1`);(a[o]==null||a[o]>i)&&e[i]===1&&(n.push(e[i]),s.push(i)),a[o]<=i&&o++}e[i]!==1&&(n.push(e[i]),s.push(i))}return{newShape:n,keptDims:s}}function or(e,t){let n=null;if(e==null||e==="float32")n=new Float32Array(t);else if(e==="int32")n=new Int32Array(t);else if(e==="bool")n=new Uint8Array(t);else throw new Error(`Unknown data type ${e}`);return n}function ir(e,t){let n=null;if(e==null||e==="float32")n=new Float32Array(t);else if(e==="int32")n=new Int32Array(t);else if(e==="bool")n=new Uint8Array(t);else if(e==="string")n=new Array(t);else throw new Error(`Unknown data type ${e}`);return n}function ur(e,t){for(let n=0;n<e.length;n++){const s=e[n];if(isNaN(s)||!isFinite(s))throw Error(`A tensor of type ${t} being uploaded contains ${s}.`)}}function cr(e){return e==="bool"||e==="complex64"||e==="float32"||e==="int32"||e==="string"}function eo(e,t){return!(t==="complex64"||t==="float32"&&e!=="complex64"||t==="int32"&&e!=="float32"&&e!=="complex64"||t==="bool"&&e==="bool")}function yt(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}function $n(e){if(e==="float32"||e==="int32")return 4;if(e==="complex64")return 8;if(e==="bool")return 1;throw new Error(`Unknown dtype ${e}`)}function lr(e){if(e==null)return 0;let t=0;return e.forEach(n=>t+=n.length),t}function en(e){return typeof e=="string"||e instanceof String}function pr(e){return typeof e=="boolean"}function hr(e){return typeof e=="number"}function nn(e){return Array.isArray(e)?nn(e[0]):e instanceof Float32Array?"float32":e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?"int32":hr(e)?"float32":en(e)?"string":pr(e)?"bool":"float32"}function Ft(e){return!!(e&&e.constructor&&e.call&&e.apply)}function no(e,t){for(let n=t;n<e;++n)if(e%n===0)return n;return e}function Fe(e){const t=e.length;if(t<2)return[];const n=new Array(t-1);n[t-2]=e[t-1];for(let s=t-3;s>=0;--s)n[s]=n[s+1]*e[s+1];return n}function fr(e,t,n,s=!1){const r=new Array;if(t.length===1){const a=t[0]*(s?2:1);for(let o=0;o<a;o++)r[o]=n[e+o]}else{const a=t[0],o=t.slice(1),i=o.reduce((u,c)=>u*c)*(s?2:1);for(let u=0;u<a;u++)r[u]=fr(e+u*i,o,n,s)}return r}function Ht(e,t,n=!1){if(e.length===0)return t[0];const s=e.reduce((r,a)=>r*a)*(n?2:1);if(s===0)return[];if(s!==t.length)throw new Error(`[${e}] does not match the input size ${t.length}${n?" for a complex tensor":""}.`);return fr(0,e,t,n)}function ss(e,t){const n=sn(e,t);for(let s=0;s<n.length;s++)n[s]=1;return n}function sn(e,t){if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool")return new Uint8Array(e);throw new Error(`Unknown data type ${t}`)}function so(e,t){const n=e.reduce((s,r)=>s*r,1);if(t==null||t==="float32")return Ht(e,new Float32Array(n));if(t==="int32")return Ht(e,new Int32Array(n));if(t==="bool")return Ht(e,new Uint8Array(n));throw new Error(`Unknown data type ${t}`)}function rs(e){e.forEach(t=>{d(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function ro(e,t,n){if(t===0)return 0;if(t===1)return e[0];let s=e[e.length-1];for(let r=0;r<e.length-1;++r)s+=n[r]*e[r];return s}function ao(e,t,n){if(t===0)return[];if(t===1)return[e];const s=new Array(t);for(let r=0;r<s.length-1;++r)s[r]=Math.floor(e/n[r]),e-=s[r]*n[r];return s[s.length-1]=e,s}function Mt(e){return e&&e.then&&typeof e.then=="function"}/**
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
 */const Cs="tfjsflags";class oo{constructor(t){this.global=t,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=io,this.populateURLFlags()}setPlatform(t,n){this.platform!=null&&(x().getBool("IS_TEST")||x().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${t}.`)),this.platformName=t,this.platform=n}registerFlag(t,n,s){if(this.flagRegistry[t]={evaluationFn:n,setHook:s},this.urlFlags[t]!=null){const r=this.urlFlags[t];x().getBool("IS_TEST")||x().getBool("PROD")||console.warn(`Setting feature override from URL ${t}: ${r}.`),this.set(t,r)}}async getAsync(t){return t in this.flags?this.flags[t]:(this.flags[t]=await this.evaluateFlag(t),this.flags[t])}get(t){if(t in this.flags)return this.flags[t];const n=this.evaluateFlag(t);if(Mt(n))throw new Error(`Flag ${t} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[t]=n,this.flags[t]}getNumber(t){return this.get(t)}getBool(t){return this.get(t)}getFlags(){return this.flags}get features(){return this.flags}set(t,n){if(this.flagRegistry[t]==null)throw new Error(`Cannot set flag ${t} as it has not been registered.`);this.flags[t]=n,this.flagRegistry[t].setHook!=null&&this.flagRegistry[t].setHook(n)}evaluateFlag(t){if(this.flagRegistry[t]==null)throw new Error(`Cannot evaluate flag '${t}': no evaluation function found.`);return this.flagRegistry[t].evaluationFn()}setFlags(t){this.flags=Object.assign({},t)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const t=this.getQueryParams(this.global.location.search);Cs in t&&t[Cs].split(",").forEach(s=>{const[r,a]=s.split(":");this.urlFlags[r]=co(r,a)})}}function io(e){const t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(n,...s)=>(uo(t,s[0],s[1]),s.join("="))),t}function uo(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||"")}function co(e,t){if(t=t.toLowerCase(),t==="true"||t==="false")return t==="true";if(`${+t}`===t)return+t;throw new Error(`Could not parse value flag value ${t} for flag ${e}.`)}function x(){return mr}let mr=null;function lo(e){mr=e}/**
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
 */let yn;function dr(){if(yn==null){let e;if(typeof window<"u")e=window;else if(typeof global<"u")e=global;else if(typeof process<"u")e=process;else if(typeof self<"u")e=self;else throw new Error("Could not find a global object");yn=e}return yn}function po(){const e=dr();return e._tfGlobals==null&&(e._tfGlobals=new Map),e._tfGlobals}function as(e,t){const n=po();if(n.has(e))return n.get(e);{const s=t();return n.set(e,s),n.get(e)}}const ho="Abs",fo="Acos",mo="Acosh",gr="Add",go="AddN",yo="All",bo="Any",wo="ArgMax",No="ArgMin",To="Asin",So="Asinh",ko="Atan",$o="Atanh",Eo="Atan2",_o="AvgPool",Bw="AvgPoolGrad",vo="AvgPool3D",Pw="AvgPool3DGrad",Io="BatchMatMul",xo="BatchToSpaceND",Do="Bincount",Lw="BroadcastTo",Ao="BroadcastArgs",yr="Cast",Oo="Ceil",Co="ClipByValue",Fo="Complex",Bo="ComplexAbs",Po="Concat",Lo="Conv2D",Ro="Conv2DBackpropFilter",zo="Conv2DBackpropInput",Vo="Conv3D",Rw="Conv3DBackpropFilterV2",jo="Conv3DBackpropInputV2",Ko="Cos",Uo="Cosh",Wo="Cumprod",qo="Cumsum",Ho="CropAndResize",Go="DenseBincount",Mo="DepthToSpace",Yo="DepthwiseConv2dNative",Xo="DepthwiseConv2dNativeBackpropFilter",Jo="DepthwiseConv2dNativeBackpropInput",Zo="Diag",Qo="Dilation2D",zw="Dilation2DBackpropInput",Vw="Dilation2DBackpropFilter",ti="RealDiv",ei="Einsum",ni="Elu",jw="EluGrad",si="Erf",ri="Equal",ai="Exp",oi="ExpandDims",ii="Expm1",ui="FFT",ci="Fill",li="FlipLeftRight",pi="Floor",hi="FloorDiv",fi="FusedBatchNorm",mi="GatherV2",di="GatherNd",gi="Greater",yi="GreaterEqual",br="Identity",bi="IFFT",wi="Imag",Ni="IsFinite",Ti="IsInf",Si="IsNan",ki="LeakyRelu",$i="Less",Ei="LessEqual",_i="LinSpace",vi="Log",Ii="Log1p",xi="LogicalAnd",Di="LogicalNot",Ai="LogicalOr",Kw="LogicalXor",Uw="LogSoftmax",Ww="LowerBound",Oi="LRN",qw="LRNGrad",Ci="Max",Fi="Maximum",Bi="MaxPool",Hw="MaxPoolGrad",Pi="MaxPool3D",Gw="MaxPool3DGrad",Li="MaxPoolWithArgmax",Ri="Mean",zi="Min",Vi="Minimum",ji="MirrorPad",Ki="Mod",Ui="Multinomial",Wi="Multiply",qi="Neg",Hi="NotEqual",Gi="NonMaxSuppressionV3",Mi="NonMaxSuppressionV4",Yi="NonMaxSuppressionV5",Xi="OnesLike",Ji="OneHot",Zi="Pack",Qi="PadV2",Mw="Pool",tu="Pow",eu="Prelu",nu="Prod",su="RaggedGather",ru="RaggedTensorToTensor",au="Range",ou="Real",iu="Reciprocal",uu="Relu",cu="Reshape",lu="ResizeNearestNeighbor",Yw="ResizeNearestNeighborGrad",pu="ResizeBilinear",Xw="ResizeBilinearGrad",hu="Relu6",fu="Reverse",mu="Round",du="Rsqrt",gu="ScatterNd",yu="SearchSorted",bu="Select",wu="Selu",Nu="Slice",Tu="Sin",Su="Sinh",ku="Sign",$u="Sigmoid",Eu="Softplus",_u="Sqrt",vu="Sum",Iu="SpaceToBatchND",xu="SplitV",Du="Softmax",Au="SparseFillEmptyRows",Ou="SparseReshape",Cu="SparseSegmentMean",Fu="SparseSegmentSum",Bu="SparseToDense",Pu="SquaredDifference",Jw="Square",Lu="StridedSlice",Ru="StringNGrams",zu="StringSplit",Vu="StringToHashBucketFast",ju="Sub",Ku="Tan",Uu="Tanh",wr="Tile",Wu="TopK",qu="Transform",bn="Transpose",Hu="Unique",Gu="Unpack",Mu="UnsortedSegmentSum",Zw="UpperBound",Yu="ZerosLike",Xu="Step",Fs="FromPixels",Ju="RotateWithOffset",Bs="_FusedMatMul",Ps="FusedConv2D",Ls="FusedDepthwiseConv2D";/**
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
 */function Kt(...e){x().getBool("IS_TEST")||x().getBool("PROD")||console.warn(...e)}function Qw(...e){x().getBool("IS_TEST")||x().getBool("PROD")||console.log(...e)}/**
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
 */const fe=as("kernelRegistry",()=>new Map),Ee=as("gradRegistry",()=>new Map);function En(e,t){const n=os(e,t);return fe.get(n)}function Rs(e){return Ee.get(e)}function _n(e){const t=fe.entries(),n=[];for(;;){const{done:s,value:r}=t.next();if(s)break;const[a,o]=r,[i]=a.split("_");i===e&&n.push(o)}return n}function Zu(e){const{kernelName:t,backendName:n}=e,s=os(t,n);fe.has(s)&&Kt(`The kernel '${t}' for backend '${n}' is already registered`),fe.set(s,e)}function tN(e){const{kernelName:t}=e;Ee.has(t)&&x().getBool("DEBUG")&&Kt(`Overriding the gradient for '${t}'`),Ee.set(t,e)}function eN(e,t){const n=os(e,t);if(!fe.has(n))throw new Error(`The kernel '${e}' for backend '${t}' is not registered`);fe.delete(n)}function nN(e){if(!Ee.has(e))throw new Error(`The gradient '${e}' for backend is not registered`);Ee.delete(e)}function sN(e,t){_n(e).forEach(s=>{const r=Object.assign({},s,{backendName:t});Zu(r)})}function os(e,t){return`${t}_${e}`}/**
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
 */const Ut=Ra||za;function Be(e){return Ut.fromString(e,!0,16)}const Nr=Be("c3a5c85c97cb3127"),Vt=Be("b492b66fbe98f273"),et=Be("9ae16a3b2f90404f");function vn(e){return e.xor(e.shru(47))}function Tr(e,t,n){const s=e.slice(t,t+n);return Ut.fromBytes(Array.from(s),!0,!0)}function L(e,t){return Tr(e,t,8)}function zs(e,t){return Tr(e,t,4)}function M(e,t){return t===0?e:e.shru(t).or(e.shl(64-t))}function Ot(e,t,n=Be("9ddfea08eb382d69")){let s=e.xor(t).mul(n);s=s.xor(s.shru(47));let r=t.xor(s).mul(n);return r=r.xor(r.shru(47)),r=r.mul(n),r}function Qu(e,t,n,s,r,a){r=r.add(e),a=M(a.add(r).add(s),21);const o=r;return r=r.add(t),r=r.add(n),a=a.add(M(r,44)),[r.add(s),a.add(o)]}function Re(e,t,n,s){return Qu(L(e,t),L(e,t+8),L(e,t+16),L(e,t+24),n,s)}function tc(e,t=e.length){if(t>=8){const n=et.add(t*2),s=L(e,0).add(et),r=L(e,t-8),a=M(r,37).mul(n).add(s),o=M(s,25).add(r).mul(n);return Ot(a,o,n)}if(t>=4){const n=et.add(t*2),s=zs(e,0);return Ot(s.shl(3).add(t),zs(e,t-4),n)}if(t>0){const n=e[0],s=e[t>>1],r=e[t-1],a=n+(s<<8),o=t+(r<<2);return vn(et.mul(a).xor(Nr.mul(o))).mul(et)}return et}function ec(e,t=e.length){const n=et.add(t*2),s=L(e,0).mul(Vt),r=L(e,8),a=L(e,t-8).mul(n),o=L(e,t-16).mul(et);return Ot(M(s.add(r),43).add(M(a,30)).add(o),s.add(M(r.add(et),18)).add(a),n)}function nc(e,t=e.length){const n=et.add(t*2),s=L(e,0).mul(et),r=L(e,8),a=L(e,t-8).mul(n),o=L(e,t-16).mul(et),i=M(s.add(r),43).add(M(a,30)).add(o),u=Ot(i,s.add(M(r.add(et),18)).add(a),n),c=L(e,16).mul(n),p=L(e,24),h=i.add(L(e,t-32)).mul(n),m=u.add(L(e,t-24)).mul(n);return Ot(M(c.add(p),43).add(M(h,30)).add(m),c.add(M(p.add(s),18)).add(h),n)}function sc(e,t=e.length){const n=Ut.fromNumber(81,!0);if(t<=32)return t<=16?tc(e,t):ec(e,t);if(t<=64)return nc(e,t);let s=n,r=n.mul(Vt).add(113),a=vn(r.mul(et).add(113)).mul(et),o=[Ut.UZERO,Ut.UZERO],i=[Ut.UZERO,Ut.UZERO];s=s.mul(et).add(L(e,0));let u=0;const c=(t-1>>6)*64,p=c+(t-1&63)-63;do s=M(s.add(r).add(o[0]).add(L(e,u+8)),37).mul(Vt),r=M(r.add(o[1]).add(L(e,u+48)),42).mul(Vt),s=s.xor(i[1]),r=r.add(o[0]).add(L(e,u+40)),a=M(a.add(i[0]),33).mul(Vt),o=Re(e,u,o[1].mul(Vt),s.add(i[0])),i=Re(e,u+32,a.add(i[1]),r.add(L(e,u+16))),[a,s]=[s,a],u+=64;while(u!==c);const h=Vt.add(a.and(255).shl(1));return u=p,i[0]=i[0].add(t-1&63),o[0]=o[0].add(i[0]),i[0]=i[0].add(o[0]),s=M(s.add(r).add(o[0]).add(L(e,u+8)),37).mul(h),r=M(r.add(o[1]).add(L(e,u+48)),42).mul(h),s=s.xor(i[1].mul(9)),r=r.add(o[0].mul(9).add(L(e,u+40))),a=M(a.add(i[0]),33).mul(h),o=Re(e,u,o[1].mul(h),s.add(i[0])),i=Re(e,u+32,a.add(i[1]),r.add(L(e,u+16))),[a,s]=[s,a],Ot(Ot(o[0],i[0],h).add(vn(r).mul(Nr)).add(a),Ot(o[1],i[1],h).add(s),h)}/**
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
 */function rc(e,t){return t==="string"?is(e):rn([e],t)}function ac(e,t){return e instanceof Float32Array&&t==="float32"||e instanceof Int32Array&&t==="int32"||e instanceof Uint8Array&&t==="bool"}function rn(e,t){if(t==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(e)&&(e=Oe(e)),x().getBool("DEBUG")&&ur(e,t),ac(e,t))return e;if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool"){const n=new Uint8Array(e.length);for(let s=0;s<n.length;++s)Math.round(e[s])!==0&&(n[s]=1);return n}else throw new Error(`Unknown data type ${t}`)}function _e(){return x().platform.now()}function oc(e,t){return x().platform.fetch(e,t)}function is(e,t="utf-8"){return t=t||"utf-8",x().platform.encode(e,t)}function In(e,t="utf-8"){return t=t||"utf-8",x().platform.decode(e,t)}const rN=Object.freeze(Object.defineProperty({__proto__:null,arraysEqual:Et,assert:d,assertNonNegativeIntegerDimensions:rs,assertNonNull:ee,assertShapesMatch:at,bytesFromStringArray:lr,bytesPerElement:$n,checkConversionForErrors:ur,clamp:Wa,computeStrides:Fe,createScalarValue:rc,createShuffledIndices:Za,decodeString:In,distSquared:Ma,encodeString:is,fetch:oc,fingerPrint64:sc,flatten:Oe,getArrayFromDType:ir,getTypedArrayFromDType:or,hasEncodingLoss:eo,hexToLong:Be,indexToLoc:ao,inferDtype:nn,inferFromImplicitShape:to,isBoolean:pr,isFunction:Ft,isInt:he,isNumber:hr,isPromise:Mt,isScalarShape:Ya,isString:en,isTypedArray:yt,isValidDtype:cr,locToIndex:ro,makeOnesTypedArray:ss,makeZerosNestedTypedArray:so,makeZerosTypedArray:sn,nearestDivisor:no,nearestLargerEven:qa,now:_e,parseAxisParam:Ce,randUniform:Ga,repeatedTry:Qa,rightPad:Se,shuffle:rr,shuffleCombo:Ua,sizeFromShape:q,sizeToSquarishShape:Ja,squeezeShape:ar,sum:Ha,swap:We,tanh:Xa,toNestedArray:Ht,toTypedArray:rn},Symbol.toStringTag,{value:"Module"}));/**
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
 */class ic{constructor(t,n){this.backendTimer=t,this.logger=n,n==null&&(this.logger=new cc)}profileKernel(t,n,s){let r;const a=()=>{r=s()};let o;const i=_e();if(this.backendTimer.timerAvailable())o=this.backendTimer.time(a);else{a();for(const c of r)c.dataSync();o=Promise.resolve({kernelMs:_e()-i})}if(x().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let c=0;c<r.length;c++){const p=r[c];p.data().then(h=>{uc(h,p.dtype,t)})}return{kernelName:t,outputs:r,inputs:n,timeMs:o.then(c=>c.kernelMs),extraInfo:o.then(c=>c.getExtraProfileInfo!=null?c.getExtraProfileInfo():"")}}logKernelProfile(t){const{kernelName:n,outputs:s,timeMs:r,inputs:a,extraInfo:o}=t;s.forEach(i=>{Promise.all([i.data(),r,o]).then(u=>{this.logger.logKernelProfile(n,i,u[0],u[1],a,u[2])})})}}function uc(e,t,n){if(t!=="float32")return!1;for(let s=0;s<e.length;s++){const r=e[s];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${n}'`),!0}return!1}class cc{logKernelProfile(t,n,s,r,a,o){const i=typeof r=="number"?Se(`${r}ms`,9):r.error,u=Se(t,25),c=n.rank,p=n.size,h=Se(n.shape.toString(),14);let m="";for(const y in a){const T=a[y];if(T!=null){const N=T.shape||n.shape,w=N.length;m+=`${y}: ${w}D ${w>0?N:""} `}}console.log(`%c${u}	%c${i}	%c${c}D ${h}	%c${p}	%c${m}	%c${o}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
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
 */function lc(e,t,n){const s={},r={};for(let u=0;u<t.length;u++)s[t[u].id]=!0;for(let u=0;u<e.length;u++){const c=e[u],p=c.inputs;for(const h in p){const m=p[h];let y=!1;for(let T=0;T<t.length;T++)if(s[m.id]){c.outputs.forEach(N=>s[N.id]=!0),y=!0,r[c.id]=!0;break}if(y)break}}const a={};a[n.id]=!0;const o={};for(let u=e.length-1;u>=0;u--){const c=e[u],p=c.inputs;for(let h=0;h<c.outputs.length;h++)if(a[c.outputs[h].id]){for(const m in p)a[p[m].id]=!0,o[c.id]=!0;break}}const i=[];for(let u=0;u<e.length;u++){const c=e[u];if(r[c.id]&&o[c.id]){const p={};for(const m in c.inputs){const y=c.inputs[m];s[y.id]&&(p[m]=y)}const h=Object.assign({},c);h.inputs=p,h.outputs=c.outputs,i.push(h)}}return i}function pc(e,t,n,s){for(let r=t.length-1;r>=0;r--){const a=t[r],o=[];if(a.outputs.forEach(u=>{const c=e[u.id];c!=null?o.push(c):o.push(null)}),a.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${a.kernelName}.`);const i=a.gradient(o);for(const u in a.inputs){if(!(u in i))throw new Error(`Cannot backprop through input ${u}. Available gradients found: ${Object.keys(i)}.`);const c=n(()=>i[u]());if(c.dtype!=="float32")throw new Error(`Error in gradient for op ${a.kernelName}. The gradient of input ${u} must have 'float32' dtype, but has '${c.dtype}'`);const p=a.inputs[u];if(!Et(c.shape,p.shape))throw new Error(`Error in gradient for op ${a.kernelName}. The gradient of input '${u}' has shape '${c.shape}', which does not match the shape of the input '${p.shape}'`);if(e[p.id]==null)e[p.id]=c;else{const h=e[p.id];e[p.id]=s(h,c),h.dispose()}}}}/**
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
 */const Vs=20,be=3,wn=7;function hc(e,t,n,s){const r=Fe(t),a=fc(e,t,n,r),o=t.length,i=Ve(e,t,n,r,a),u=["Tensor"];return s&&(u.push(`  dtype: ${n}`),u.push(`  rank: ${o}`),u.push(`  shape: [${t}]`),u.push("  values:")),u.push(i.map(c=>"    "+c).join(`
`)),u.join(`
`)}function fc(e,t,n,s){const r=q(t),a=s[s.length-1],o=new Array(a).fill(0),i=t.length,u=n==="complex64"?Te(e):e;if(i>1)for(let c=0;c<r/a;c++){const p=c*a;for(let h=0;h<a;h++)o[h]=Math.max(o[h],Ne(u[p+h],0,n).length)}return o}function Ne(e,t,n){let s;return Array.isArray(e)?s=`${parseFloat(e[0].toFixed(wn))} + ${parseFloat(e[1].toFixed(wn))}j`:en(e)?s=`'${e}'`:n==="bool"?s=Sr(e):s=parseFloat(e.toFixed(wn)).toString(),Se(s,t)}function Sr(e){return e===0?"false":"true"}function Ve(e,t,n,s,r,a=!0){const o=n==="complex64"?2:1,i=t[0],u=t.length;if(u===0){if(n==="complex64"){const N=Te(e);return[Ne(N[0],0,n)]}return n==="bool"?[Sr(e[0])]:[e[0].toString()]}if(u===1){if(i>Vs){const w=be*o;let k=Array.from(e.slice(0,w)),v=Array.from(e.slice((i-be)*o,i*o));return n==="complex64"&&(k=Te(k),v=Te(v)),["["+k.map(($,E)=>Ne($,r[E],n)).join(", ")+", ..., "+v.map(($,E)=>Ne($,r[i-be+E],n)).join(", ")+"]"]}return["["+(n==="complex64"?Te(e):Array.from(e)).map((w,k)=>Ne(w,r[k],n)).join(", ")+"]"]}const c=t.slice(1),p=s.slice(1),h=s[0]*o,m=[];if(i>Vs){for(let N=0;N<be;N++){const w=N*h,k=w+h;m.push(...Ve(e.slice(w,k),c,n,p,r,!1))}m.push("...");for(let N=i-be;N<i;N++){const w=N*h,k=w+h;m.push(...Ve(e.slice(w,k),c,n,p,r,N===i-1))}}else for(let N=0;N<i;N++){const w=N*h,k=w+h;m.push(...Ve(e.slice(w,k),c,n,p,r,N===i-1))}const y=u===2?",":"";m[0]="["+m[0]+y;for(let N=1;N<m.length-1;N++)m[N]=" "+m[N]+y;let T=`,
`;for(let N=2;N<u;N++)T+=`
`;return m[m.length-1]=" "+m[m.length-1]+"]"+(a?"":T),m}function Te(e){const t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}/**
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
 */class xn{constructor(t,n,s){if(this.dtype=n,this.shape=t.slice(),this.size=q(t),s!=null){const r=s.length;d(r===this.size,()=>`Length of values '${r}' does not match the size inferred by the shape '${this.size}'.`)}if(n==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=s||ir(n,this.size),this.strides=Fe(t)}set(t,...n){n.length===0&&(n=[0]),d(n.length===this.rank,()=>`The number of provided coordinates (${n.length}) must match the rank (${this.rank})`);const s=this.locToIndex(n);this.values[s]=t}get(...t){t.length===0&&(t=[0]);let n=0;for(const r of t){if(r<0||r>=this.shape[n]){const a=`Requested out of range element at ${t}.   Buffer shape=${this.shape}`;throw new Error(a)}n++}let s=t[t.length-1];for(let r=0;r<t.length-1;++r)s+=this.strides[r]*t[r];return this.values[s]}locToIndex(t){if(this.rank===0)return 0;if(this.rank===1)return t[0];let n=t[t.length-1];for(let s=0;s<t.length-1;++s)n+=this.strides[s]*t[s];return n}indexToLoc(t){if(this.rank===0)return[];if(this.rank===1)return[t];const n=new Array(this.shape.length);for(let s=0;s<n.length-1;++s)n[s]=Math.floor(t/this.strides[s]),t-=n[s]*this.strides[s];return n[n.length-1]=t,n}get rank(){return this.shape.length}toTensor(){return mt().makeTensor(this.values,this.shape,this.dtype)}}let mt=null,oe=null;function mc(e){mt=e}function dc(e){oe=e}class W{constructor(t,n,s,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=t.slice(),this.dtype=n||"float32",this.size=q(t),this.strides=Fe(t),this.dataId=s,this.id=r,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const t=await this.data();return oe.buffer(this.shape,this.dtype,t)}bufferSync(){return oe.buffer(this.shape,this.dtype,this.dataSync())}async array(){const t=await this.data();return Ht(this.shape,t,this.dtype==="complex64")}arraySync(){return Ht(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const t=mt().read(this.dataId);if(this.dtype==="string"){const n=await t;try{return n.map(s=>In(s))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return t}dataToGPU(t){return this.throwIfDisposed(),mt().readToGPU(this.dataId,t)}dataSync(){this.throwIfDisposed();const t=mt().readSync(this.dataId);if(this.dtype==="string")try{return t.map(n=>In(n))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return t}async bytes(){this.throwIfDisposed();const t=await mt().read(this.dataId);return this.dtype==="string"?t:new Uint8Array(t.buffer)}dispose(){this.isDisposed||(mt().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(t=!1){return oe.print(this,t)}clone(){return this.throwIfDisposed(),oe.clone(this)}toString(t=!1){const n=this.dataSync();return hc(n,this.shape,this.dtype,t)}cast(t){return this.throwIfDisposed(),oe.cast(this,t)}variable(t=!0,n,s){return this.throwIfDisposed(),mt().makeVariable(this,t,n,s)}}Object.defineProperty(W,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function gc(){return as("Tensor",()=>W)}gc();class qe extends W{constructor(t,n,s,r){super(t.shape,t.dtype,t.dataId,r),this.trainable=n,this.name=s}assign(t){if(t.dtype!==this.dtype)throw new Error(`dtype of the new value (${t.dtype}) and previous value (${this.dtype}) must match`);if(!Et(t.shape,this.shape))throw new Error(`shape of the new value (${t.shape}) and previous value (${this.shape}) must match`);mt().disposeTensor(this),this.dataId=t.dataId,mt().incRef(this,null)}dispose(){mt().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(qe,Symbol.hasInstance,{value:e=>e instanceof W&&e.assign!=null&&e.assign instanceof Function});/**
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
 */var js;(function(e){e.R0="R0",e.R1="R1",e.R2="R2",e.R3="R3",e.R4="R4",e.R5="R5",e.R6="R6"})(js||(js={}));var Dn;(function(e){e.float32="float32",e.int32="int32",e.bool="int32",e.complex64="complex64"})(Dn||(Dn={}));var An;(function(e){e.float32="float32",e.int32="int32",e.bool="bool",e.complex64="complex64"})(An||(An={}));var On;(function(e){e.float32="float32",e.int32="float32",e.bool="float32",e.complex64="complex64"})(On||(On={}));var Cn;(function(e){e.float32="complex64",e.int32="complex64",e.bool="complex64",e.complex64="complex64"})(Cn||(Cn={}));const yc={float32:On,int32:Dn,bool:An,complex64:Cn};function kr(e,t){if(e==="string"||t==="string"){if(e==="string"&&t==="string")return"string";throw new Error(`Can not upcast ${e} with ${t}`)}return yc[e][t]}function aN(e){return kr(e,"int32")}/**
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
 */function U(e,t){if(e.dtype===t.dtype)return[e,t];const n=kr(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function $r(e,t){d(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function bc(e,t){return t.some(n=>n.id===e.id)}function us(e){const t=[];return Er(e,t,new Set),t}function Er(e,t,n){if(e==null)return;if(e instanceof W){t.push(e);return}if(!wc(e))return;const s=e;for(const r in s){const a=s[r];n.has(a)||(n.add(a),Er(a,t,n))}}function wc(e){return Array.isArray(e)||typeof e=="object"}const oN=Object.freeze(Object.defineProperty({__proto__:null,assertTypesMatch:$r,getTensorsInContainer:us,isTensorInList:bc,makeTypesMatch:U},Symbol.toStringTag,{value:"Module"}));/**
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
 */function Nn(e){return e.kernelName!=null}class Ks{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(t=>t.name)))}}}dispose(){for(const t in this.registeredVariables)this.registeredVariables[t].dispose()}}class me{constructor(t){this.ENV=t,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Ks}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const s=t[n];if(await this.initializeBackend(s).success){await this.setBackend(s);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:t,asyncInit:n}=this.initializeBackendsAndReturnBest();if(n)throw new Error(`The highest priority backend '${t}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(t)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(t){if(!(t in this.registry))if(t in this.registryFactory){const{asyncInit:n}=this.initializeBackend(t);if(n)return null}else return null;return this.registry[t]}findBackendFactory(t){return t in this.registryFactory?this.registryFactory[t].factory:null}registerBackend(t,n,s=1){return t in this.registryFactory?(Kt(`${t} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[t]={factory:n,priority:s},!0)}async setBackend(t){if(this.registryFactory[t]==null)throw new Error(`Backend name '${t}' not found in registry`);if(this.backendName=t,this.registry[t]==null){this.backendInstance=null;const{success:n,asyncInit:s}=this.initializeBackend(t);if(!(s?await n:n))return!1}return this.backendInstance=this.registry[t],this.setupRegisteredKernels(),this.profiler=new ic(this.backendInstance),!0}setupRegisteredKernels(){_n(this.backendName).forEach(n=>{n.setupFunc!=null&&n.setupFunc(this.backendInstance)})}disposeRegisteredKernels(t){_n(t).forEach(s=>{s.disposeFunc!=null&&s.disposeFunc(this.registry[t])})}initializeBackend(t){const n=this.registryFactory[t];if(n==null)throw new Error(`Cannot initialize backend ${t}, no registration found.`);try{const s=n.factory();if(s&&!(s instanceof Ka)&&typeof s.then=="function"){const r=++this.pendingBackendInitId,a=s.then(o=>r<this.pendingBackendInitId?!1:(this.registry[t]=o,this.pendingBackendInit=null,!0)).catch(o=>(r<this.pendingBackendInitId||(this.pendingBackendInit=null,Kt(`Initialization of backend ${t} failed`),Kt(o.stack||o.message)),!1));return this.pendingBackendInit=a,{success:a,asyncInit:!0}}else return this.registry[t]=s,{success:!0,asyncInit:!1}}catch(s){return Kt(`Initialization of backend ${t} failed`),Kt(s.stack||s.message),{success:!1,asyncInit:!1}}}removeBackend(t){if(!(t in this.registryFactory))throw new Error(`${t} backend not found in registry`);this.backendName===t&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,t in this.registry&&(this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t]),delete this.registryFactory[t],this.backendName===t&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((t,n)=>this.registryFactory[n].priority-this.registryFactory[t].priority)}initializeBackendsAndReturnBest(){const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const s=t[n],{success:r,asyncInit:a}=this.initializeBackend(s);if(a||r)return{name:s,asyncInit:a}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(t,n){const s=this.state.tensorInfo.get(n),r=s.backend,a=this.readSync(n),o=r.refCount(n);r.disposeData(n,!0),s.backend=t,t.move(n,a,s.shape,s.dtype,o),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(t,n){let s=null;if(n==null){if(typeof t!="function")throw new Error("Please provide a function to tidy()");n=t}else{if(typeof t!="string"&&!(t instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof n!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");s=t}let r;return this.scopedRun(()=>this.startScope(s),()=>this.endScope(r),()=>(r=n(),r instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),r))}scopedRun(t,n,s){t();try{const r=s();return n(),r}catch(r){throw n(),r}}nextTensorId(){return me.nextTensorId++}nextVariableId(){return me.nextVariableId++}clone(t){const n=b.runKernel(br,{x:t}),s={x:t},r=o=>({x:()=>{const i="float32",u={x:o},c={dtype:i};return b.runKernel(yr,u,c)}}),a=[];return this.addTapeNode(this.state.activeScope.name,s,[n],r,a,{}),n}runKernel(t,n,s){if(this.backendName==null&&this.backend,!(En(t,this.backendName)!=null))throw new Error(`Kernel '${t}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:t,inputs:n,attrs:s})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(t,n,s){const r=this.backend.numDataIds();let a=0;s.forEach(u=>{a+=u.dtype==="complex64"?3:1});const o=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],i=r-n-a-o;if(i>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${i} data ids) after running '${t}'`)}runKernelFunc(t){let n,s=[];const r=this.isTapeOn(),a=this.state.numBytes,o=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let i;this.backendName==null&&this.backend;let u;const c=Nn(t)?t.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(Nn(t)){const{kernelName:T,inputs:N,attrs:w}=t;this.backendName==null&&this.backend;const k=En(T,this.backendName);d(k!=null,()=>`Cannot find registered kernel '${T}' for backend '${this.backendName}'`),i=()=>{const v=this.backend.numDataIds();u=k.kernelFunc({inputs:N,attrs:w,backend:this.backend});const $=Array.isArray(u)?u:[u];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(T,v,$);const E=$.map(I=>I.rank!=null?I:this.makeTensorFromTensorInfo(I));if(r){const I=this.getTensorsForGradient(T,N,E);s=this.saveTensorsForBackwardMode(I)}return E}}else{const{forwardFunc:T}=t,N=w=>{r&&(s=w.map(k=>this.keep(this.clone(k))))};i=()=>{const w=this.backend.numDataIds();u=this.tidy(()=>T(this.backend,N));const k=Array.isArray(u)?u:[u];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(c,w,k),k}}const{inputs:p,attrs:h}=t,m=Nn(t)?null:t.backwardsFunc;let y;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?n=i():(y=this.profiler.profileKernel(c,p,()=>i()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(y),n=y.outputs)}),r&&this.addTapeNode(c,p,n,m,s,h),this.state.profiling&&this.state.activeProfile.kernels.push({name:c,bytesAdded:this.state.numBytes-a,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-o,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(p).map(T=>p[T]!=null?p[T].shape:null),outputShapes:n.map(T=>T.shape),kernelTimeMs:y.timeMs,extraInfo:y.extraInfo}),Array.isArray(u)?n:n[0]}saveTensorsForBackwardMode(t){return t.map(s=>this.keep(this.clone(s)))}getTensorsForGradient(t,n,s){const r=Rs(t);if(r!=null){const a=r.inputsToSave||[],o=r.outputsToSave||[];let i;r.saveAllInputs?(d(Array.isArray(n),()=>"saveAllInputs is true, expected inputs to be an array."),i=Object.keys(n).map(c=>n[c])):i=a.map(c=>n[c]);const u=s.filter((c,p)=>o[p]);return i.concat(u)}return[]}makeTensor(t,n,s,r){if(t==null)throw new Error("Values passed to engine.makeTensor() are null");s=s||"float32",r=r||this.backend;let a=t;s==="string"&&en(t[0])&&(a=t.map(u=>is(u)));const o=r.write(a,n,s),i=new W(n,s,o,this.nextTensorId());if(this.trackTensor(i,r),s==="string"){const u=this.state.tensorInfo.get(o),c=lr(a);this.state.numBytes+=c-u.bytes,u.bytes=c}return i}makeTensorFromDataId(t,n,s,r){s=s||"float32";const a={dataId:t,shape:n,dtype:s};return this.makeTensorFromTensorInfo(a,r)}makeTensorFromTensorInfo(t,n){const{dataId:s,shape:r,dtype:a}=t,o=new W(r,a,s,this.nextTensorId());return this.trackTensor(o,n),o}makeVariable(t,n=!0,s,r){s=s||this.nextVariableId().toString(),r!=null&&r!==t.dtype&&(t=t.cast(r));const a=new qe(t,n,s,this.nextTensorId());if(this.state.registeredVariables[a.name]!=null)throw new Error(`Variable with name ${a.name} was already registered`);return this.state.registeredVariables[a.name]=a,this.incRef(a,this.backend),a}trackTensor(t,n){this.state.numTensors++,t.dtype==="string"&&this.state.numStringTensors++;let s=0;t.dtype!=="complex64"&&t.dtype!=="string"&&(s=t.size*$n(t.dtype)),this.state.numBytes+=s,this.state.tensorInfo.has(t.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(t.dataId,{backend:n||this.backend,dtype:t.dtype,shape:t.shape,bytes:s})),t instanceof qe||this.track(t)}incRef(t,n){this.trackTensor(t,n),this.backend.incRef(t.dataId)}removeDataId(t,n){this.state.tensorInfo.has(t)&&this.state.tensorInfo.get(t).backend===n&&(this.state.tensorInfo.delete(t),this.state.numDataBuffers--)}disposeTensor(t){if(!this.state.tensorInfo.has(t.dataId))return;const n=this.state.tensorInfo.get(t.dataId);if(this.state.numTensors--,t.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=n.bytes),t.dtype!=="complex64"&&t.dtype!=="string"){const s=t.size*$n(t.dtype);this.state.numBytes-=s}n.backend.disposeData(t.dataId)&&this.removeDataId(t.dataId,n.backend)}disposeVariables(){for(const t in this.state.registeredVariables){const n=this.state.registeredVariables[t];this.disposeVariable(n)}}disposeVariable(t){this.disposeTensor(t),this.state.registeredVariables[t.name]!=null&&delete this.state.registeredVariables[t.name]}memory(){const t=this.backend.memory();return t.numTensors=this.state.numTensors,t.numDataBuffers=this.state.numDataBuffers,t.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(t.unreliable=!0,t.reasons==null&&(t.reasons=[]),t.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),t}async profile(t){this.state.profiling=!0;const n=this.state.numBytes,s=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await t(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(r=>r.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-n,this.state.activeProfile.newTensors=this.state.numTensors-s;for(const r of this.state.activeProfile.kernels)r.kernelTimeMs=await r.kernelTimeMs,r.extraInfo=await r.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(t,n,s,r,a,o){const i={id:this.state.nextTapeNodeId++,kernelName:t,inputs:n,outputs:s,saved:a},u=Rs(t);u!=null&&(r=u.gradFunc),r!=null&&(i.gradient=c=>(c=c.map((p,h)=>{if(p==null){const m=s[h],y=sn(m.size,m.dtype);return this.makeTensor(y,m.shape,m.dtype)}return p}),r(c.length>1?c:c[0],a,o))),this.state.activeTape.push(i)}keep(t){return t.kept=!0,t}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(t){const n={track:[],name:"unnamed scope",id:this.state.nextScopeId++};t&&(n.name=t),this.state.scopeStack.push(n),this.state.activeScope=n}endScope(t){const n=us(t),s=new Set(n.map(a=>a.id));for(let a=0;a<this.state.activeScope.track.length;a++){const o=this.state.activeScope.track[a];!o.kept&&!s.has(o.id)&&o.dispose()}const r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],n.forEach(a=>{!a.kept&&a.scopeId===r.id&&this.track(a)})}gradients(t,n,s,r=!1){if(d(n.length>0,()=>"gradients() received an empty list of xs."),s!=null&&s.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${s.dtype}'`);const a=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",t));d(a instanceof W,()=>"The result y returned by f() must be a tensor.");const o=lc(this.state.activeTape,n,a);if(!r&&o.length===0&&n.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const i={};i[a.id]=s??Nc(a.shape),pc(i,o,c=>this.tidy(c),Tc);const u=n.map(c=>i[c.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(c=>{for(const p of c.saved)p.dispose()}),this.state.activeTape=null),{value:a,grads:u}})}customGrad(t){return d(Ft(t),()=>"The f passed in customGrad(f) must be a function."),(...n)=>{d(n.every(i=>i instanceof W),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let s;const r={};n.forEach((i,u)=>{r[u]=i});const a=(i,u)=>(s=t(...n,u),d(s.value instanceof W,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),d(Ft(s.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),s.value),o=(i,u)=>{const c=s.gradFunc(i,u),p=Array.isArray(c)?c:[c];d(p.length===n.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),d(p.every(m=>m instanceof W),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const h={};return p.forEach((m,y)=>{h[y]=()=>m}),h};return this.runKernelFunc({forwardFunc:a,backwardsFunc:o,inputs:r})}}readSync(t){return this.state.tensorInfo.get(t).backend.readSync(t)}read(t){return this.state.tensorInfo.get(t).backend.read(t)}readToGPU(t,n){return this.state.tensorInfo.get(t).backend.readToGPU(t,n)}async time(t){const n=_e(),s=await this.backend.time(t);return s.wallMs=_e()-n,s}track(t){return this.state.activeScope!=null&&(t.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(t)),t}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Ks;for(const t in this.registry)this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}me.nextTensorId=0;me.nextVariableId=0;function Nc(e){const t=ss(q(e),"float32");return b.makeTensor(t,e,"float32")}function _r(){const e=dr();if(e._tfengine==null){const t=new oo(e);e._tfengine=new me(t)}return lo(e._tfengine.ENV),mc(()=>e._tfengine),e._tfengine}const b=_r();function Tc(e,t){const n={a:e,b:t};return b.runKernel(gr,n)}/**
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
 */function Sc(){return typeof navigator<"u"&&navigator!=null}let Fn;function kc(e){Fn=e}function $c(e){if(Fn!==void 0)return Fn;if(e||Sc()){if(e||(e=navigator),e.product==="ReactNative")return!0;const t=e.userAgent||e.vendor||(typeof window<"u"?window.opera:"");if(!t){const n=e;return n.userAgentData&&n.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function vr(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}const iN=Object.freeze(Object.defineProperty({__proto__:null,isBrowser:vr,isMobile:$c,mockIsMobile:kc},Symbol.toStringTag,{value:"Module"}));/**
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
 */const ct=x();ct.registerFlag("DEBUG",()=>!1,e=>{e&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});ct.registerFlag("IS_BROWSER",()=>vr());ct.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");ct.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));ct.registerFlag("PROD",()=>!1);ct.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>ct.getBool("DEBUG"));ct.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);ct.registerFlag("IS_TEST",()=>!1);ct.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>!0);ct.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);ct.registerFlag("ENGINE_COMPILE_ONLY",()=>!1);ct.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);ct.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
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
 */function Pt(e,t){let n=e;if(yt(e))return t==="string"?[]:[e.length];if(!Array.isArray(e))return[];const s=[];for(;Array.isArray(n)||yt(n)&&t!=="string";)s.push(n.length),n=n[0];return Array.isArray(e)&&x().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&Ir(e,s,[]),s}function Ir(e,t,n){if(n=n||[],!Array.isArray(e)&&!yt(e)){d(t.length===0,()=>`Element arr[${n.join("][")}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}d(t.length>0,()=>`Element arr[${n.join("][")}] should be a primitive, but is an array of ${e.length} elements`),d(e.length===t[0],()=>`Element arr[${n.join("][")}] should have ${t[0]} elements, but has ${e.length} elements`);const s=t.slice(1);for(let r=0;r<e.length;++r)Ir(e[r],s,n.concat(r))}function Us(e,t,n,s){if(e!=="string_or_numeric"){if(e==null)throw new Error("Expected dtype cannot be null.");if(e!=="numeric"&&e!==t||e==="numeric"&&t==="string")throw new Error(`Argument '${n}' passed to '${s}' must be ${e} tensor, but got ${t} tensor`)}}function f(e,t,n,s="numeric"){if(e instanceof W)return Us(s,e.dtype,t,n),e;let r=nn(e);if(r!=="string"&&["bool","int32","float32"].indexOf(s)>=0&&(r=s),Us(s,r,t,n),e==null||!yt(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string"){const u=e==null?"null":e.constructor.name;throw new Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${u}'`)}const a=Pt(e,r);!yt(e)&&!Array.isArray(e)&&(e=[e]);const i=r!=="string"?rn(e,r):Oe(e,[],!0);return b.makeTensor(i,a,r)}function ve(e,t,n,s="numeric"){if(!Array.isArray(e))throw new Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((a,o)=>f(a,`${t}[${o}]`,n,s))}/**
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
 */const xr="__op";function g(e){const t=Object.keys(e);if(t.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0];const s=e[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n=n+xr;const r=(...a)=>{b.startScope(n);try{const o=s(...a);return Mt(o)&&console.error("Cannot return a Promise inside of tidy."),b.endScope(o),o}catch(o){throw b.endScope(null),o}};return Object.defineProperty(r,"name",{value:n,configurable:!0}),r}/**
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
 */function Ec(e,t){const n=f(e,"real","complex"),s=f(t,"imag","complex");at(n.shape,s.shape,`real and imag shapes, ${n.shape} and ${s.shape}, must match in call to tf.complex().`);const r={real:n,imag:s};return b.runKernel(Fo,r)}const Bt=g({complex_:Ec});/**
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
 */function Lt(e,t,n,s){if(s==null&&(s=nn(e)),s==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(!yt(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(t!=null){rs(t);const r=q(t),a=q(n);d(r===a,()=>`Based on the provided shape, [${t}], the tensor should have ${r} values but has ${a}`);for(let o=0;o<n.length;++o){const i=n[o],u=o===n.length-1?i!==q(t.slice(o)):!0;d(n[o]===t[o]||!u,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!yt(e)&&!Array.isArray(e)&&(e=[e]),t=t||n,e=s!=="string"?rn(e,s):Oe(e,[],!0),b.makeTensor(e,t,s)}/**
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
 */function Nt(e,t,n){const s=Pt(e,n);return Lt(e,t,s,n)}/**
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
 */const Bn={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};/**
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
 */const He=4;async function _c(e,t){const n=[],s=[],r=Array.isArray(e)?e.map(o=>o.name):Object.keys(e);for(let o=0;o<r.length;++o){const i=r[o],u=Array.isArray(e)?e[o].tensor:e[i];if(u.dtype!=="float32"&&u.dtype!=="int32"&&u.dtype!=="bool"&&u.dtype!=="string"&&u.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${i}': ${u.dtype}`);const c={name:i,shape:u.shape,dtype:u.dtype};if(u.dtype==="string"){const p=new Promise(async h=>{const m=await u.bytes(),y=m.reduce((w,k)=>w+k.length,0)+He*m.length,T=new Uint8Array(y);let N=0;for(let w=0;w<m.length;w++){const k=m[w],v=new Uint8Array(new Uint32Array([k.length]).buffer);T.set(v,N),N+=He,T.set(k,N),N+=k.length}h(T)});s.push(p)}else s.push(u.data());t!=null&&(c.group=t),n.push(c)}const a=await Promise.all(s);return{data:vc(a),specs:n}}function Dr(e,t){const n={};let s,r=0;for(const a of t){const o=a.name,i=a.dtype,u=a.shape,c=q(u);let p;if("quantization"in a){const h=a.quantization;if(h.dtype==="uint8"||h.dtype==="uint16"){if(!("min"in h&&"scale"in h))throw new Error(`Weight ${a.name} with quantization ${h.dtype} doesn't have corresponding metadata min and scale.`)}else if(h.dtype==="float16"){if(i!=="float32")throw new Error(`Weight ${a.name} is quantized with ${h.dtype} which only supports weights of type float32 not ${i}.`)}else throw new Error(`Weight ${a.name} has unknown quantization dtype ${h.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);const m=Bn[h.dtype],y=e.slice(r,r+c*m),T=h.dtype==="uint8"?new Uint8Array(y):new Uint16Array(y);if(i==="float32")if(h.dtype==="uint8"||h.dtype==="uint16"){p=new Float32Array(T.length);for(let N=0;N<T.length;N++){const w=T[N];p[N]=w*h.scale+h.min}}else if(h.dtype==="float16")s===void 0&&(s=Cc()),p=s(T);else throw new Error(`Unsupported quantization type ${h.dtype} for weight type float32.`);else if(i==="int32"){if(h.dtype!=="uint8"&&h.dtype!=="uint16")throw new Error(`Unsupported quantization type ${h.dtype} for weight type int32.`);p=new Int32Array(T.length);for(let N=0;N<T.length;N++){const w=T[N];p[N]=Math.round(w*h.scale+h.min)}}else throw new Error(`Unsupported dtype in weight '${o}': ${i}`);r+=c*m}else if(i==="string"){const h=q(a.shape);p=[];for(let m=0;m<h;m++){const y=new Uint32Array(e.slice(r,r+He))[0];r+=He;const T=new Uint8Array(e.slice(r,r+y));p.push(T),r+=y}}else{const h=Bn[i],m=e.slice(r,r+c*h);if(i==="float32")p=new Float32Array(m);else if(i==="int32")p=new Int32Array(m);else if(i==="bool")p=new Uint8Array(m);else if(i==="complex64"){p=new Float32Array(m);const y=new Float32Array(p.length/2),T=new Float32Array(p.length/2);for(let k=0;k<y.length;k++)y[k]=p[k*2],T[k]=p[k*2+1];const N=Nt(y,u,"float32"),w=Nt(T,u,"float32");n[o]=Bt(N,w),N.dispose(),w.dispose()}else throw new Error(`Unsupported dtype in weight '${o}': ${i}`);r+=c*h}i!=="complex64"&&(n[o]=Nt(p,u,i))}return n}function vc(e){if(e===null)throw new Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0;const n=[];e.forEach(a=>{if(t+=a.byteLength,n.push(a.byteLength===a.buffer.byteLength?a:new a.constructor(a)),!(a instanceof Float32Array||a instanceof Int32Array||a instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${a.constructor.name}`)});const s=new Uint8Array(t);let r=0;return n.forEach(a=>{s.set(new Uint8Array(a.buffer),r),r+=a.byteLength}),s.buffer}const cs=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function Ws(e){return cs?Buffer.byteLength(e):new Blob([e]).size}function Ic(e){if(cs)return Buffer.from(e).toString("base64");const t=new Uint8Array(e);let n="";for(let s=0,r=t.length;s<r;s++)n+=String.fromCharCode(t[s]);return btoa(n)}function xc(e){if(cs){const s=Buffer.from(e,"base64");return s.buffer.slice(s.byteOffset,s.byteOffset+s.byteLength)}const t=atob(e),n=new Uint8Array(t.length);for(let s=0;s<t.length;++s)n.set([t.charCodeAt(s)],s);return n.buffer}function ls(e){if(e.length===1)return e[0];let t=0;e.forEach(r=>{t+=r.byteLength});const n=new Uint8Array(t);let s=0;return e.forEach(r=>{n.set(new Uint8Array(r),s),s+=r.byteLength}),n.buffer}function qs(e){const t="/";for(e=e.trim();e.endsWith(t);)e=e.slice(0,e.length-1);const n=e.split(t);return n[n.length-1]}function Ar(e,t){const n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return e.signature!=null&&(n.signature=e.signature),e.userDefinedMetadata!=null&&(n.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(n.modelInitializer=e.modelInitializer),e.trainingConfig!=null&&(n.trainingConfig=e.trainingConfig),n}function ps(e,t,n){const s={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(e.trainingConfig!=null&&(s.trainingConfig=e.trainingConfig),e.weightsManifest!=null){if(!t)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!n)throw new Error("modelJSON has weightsManifest but weightData is null");s.weightSpecs=t,s.weightData=n}return e.signature!=null&&(s.signature=e.signature),e.userDefinedMetadata!=null&&(s.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(s.modelInitializer=e.modelInitializer),s}async function hs(e,t){let n,s;return e.weightsManifest!=null&&([n,s]=await t(e.weightsManifest)),ps(e,n,s)}function Pe(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:e.modelTopology==null?0:Ws(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:Ws(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:e.weightData.byteLength}}function fs(e){const t=[];for(const n of e)t.push(...n.weights);return t}function Dc(){const e=n=>{let s=n<<13,r=0;for(;!(s&8388608);)r-=8388608,s<<=1;return s&=-8388609,r+=947912704,s|r},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let n=1024;n<2048;n++)t[n]=939524096+(n-1024<<13);return t}function Ac(){const e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}function Oc(){const e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}function Cc(){const e=Dc(),t=Ac(),n=Oc();return s=>{const r=new ArrayBuffer(4*s.length),a=new Uint32Array(r);for(let o=0;o<s.length;o++){const i=s[o],u=e[n[i>>10]+(i&1023)]+t[i>>10];a[o]=u}return new Float32Array(r)}}/**
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
 */class K{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return K.instance==null&&(K.instance=new K),K.instance}static registerSaveRouter(t){K.getInstance().saveRouters.push(t)}static registerLoadRouter(t){K.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return K.getHandlers(t,"save")}static getLoadHandlers(t,n){return K.getHandlers(t,"load",n)}static getHandlers(t,n,s){const r=[];return(n==="load"?K.getInstance().loadRouters:K.getInstance().saveRouters).forEach(o=>{const i=o(t,s);i!==null&&r.push(i)}),r}}const Fc=e=>K.registerSaveRouter(e),Bc=e=>K.registerLoadRouter(e),Pc=e=>K.getSaveHandlers(e),Lc=(e,t)=>K.getLoadHandlers(e,t);/**
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
 */const Pn="tensorflowjs",Ln=1,Wt="models_store",Dt="model_info_store";function Or(){if(!x().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const e=typeof window>"u"?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw new Error("The current browser does not appear to support IndexedDB.");return t}function Rn(e){const t=e.result;t.createObjectStore(Wt,{keyPath:"modelPath"}),t.createObjectStore(Dt,{keyPath:"modelPath"})}class Yt{constructor(t){if(this.indexedDB=Or(),t==null||!t)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=t}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,t)}async load(){return this.databaseAction(this.modelPath)}databaseAction(t,n){return new Promise((s,r)=>{const a=this.indexedDB.open(Pn,Ln);a.onupgradeneeded=()=>Rn(a),a.onsuccess=()=>{const o=a.result;if(n==null){const i=o.transaction(Wt,"readonly"),c=i.objectStore(Wt).get(this.modelPath);c.onsuccess=()=>{if(c.result==null)return o.close(),r(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));s(c.result.modelArtifacts)},c.onerror=p=>(o.close(),r(c.error)),i.oncomplete=()=>o.close()}else{const i=Pe(n),u=o.transaction(Dt,"readwrite");let c=u.objectStore(Dt);const p=c.put({modelPath:this.modelPath,modelArtifactsInfo:i});let h;p.onsuccess=()=>{h=o.transaction(Wt,"readwrite");const y=h.objectStore(Wt).put({modelPath:this.modelPath,modelArtifacts:n,modelArtifactsInfo:i});y.onsuccess=()=>s({modelArtifactsInfo:i}),y.onerror=T=>{c=u.objectStore(Dt);const N=c.delete(this.modelPath);N.onsuccess=()=>(o.close(),r(y.error)),N.onerror=w=>(o.close(),r(y.error))}},p.onerror=m=>(o.close(),r(p.error)),u.oncomplete=()=>{h==null?o.close():h.oncomplete=()=>o.close()}}},a.onerror=o=>r(a.error)})}}Yt.URL_SCHEME="indexeddb://";const Cr=e=>x().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Yt.URL_SCHEME)?Rc(e.slice(Yt.URL_SCHEME.length)):null;K.registerSaveRouter(Cr);K.registerLoadRouter(Cr);function Rc(e){return new Yt(e)}function zc(e){return e.startsWith(Yt.URL_SCHEME)?e.slice(Yt.URL_SCHEME.length):e}class Vc{constructor(){this.indexedDB=Or()}async listModels(){return new Promise((t,n)=>{const s=this.indexedDB.open(Pn,Ln);s.onupgradeneeded=()=>Rn(s),s.onsuccess=()=>{const r=s.result,a=r.transaction(Dt,"readonly"),i=a.objectStore(Dt).getAll();i.onsuccess=()=>{const u={};for(const c of i.result)u[c.modelPath]=c.modelArtifactsInfo;t(u)},i.onerror=u=>(r.close(),n(i.error)),a.oncomplete=()=>r.close()},s.onerror=r=>n(s.error)})}async removeModel(t){return t=zc(t),new Promise((n,s)=>{const r=this.indexedDB.open(Pn,Ln);r.onupgradeneeded=()=>Rn(r),r.onsuccess=()=>{const a=r.result,o=a.transaction(Dt,"readwrite"),i=o.objectStore(Dt),u=i.get(t);let c;u.onsuccess=()=>{if(u.result==null)return a.close(),s(new Error(`Cannot find model with path '${t}' in IndexedDB.`));{const p=i.delete(t),h=()=>{c=a.transaction(Wt,"readwrite");const y=c.objectStore(Wt).delete(t);y.onsuccess=()=>n(u.result.modelArtifactsInfo),y.onerror=T=>s(u.error)};p.onsuccess=h,p.onerror=m=>(h(),a.close(),s(u.error))}},u.onerror=p=>(a.close(),s(u.error)),o.oncomplete=()=>{c==null?a.close():c.oncomplete=()=>a.close()}},r.onerror=a=>s(r.error)})}}/**
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
 */const kt="/",ie="tensorflowjs_models",Fr="info",jc="model_topology",Kc="weight_specs",Uc="weight_data",Wc="model_metadata";function Br(e){return{info:[ie,e,Fr].join(kt),topology:[ie,e,jc].join(kt),weightSpecs:[ie,e,Kc].join(kt),weightData:[ie,e,Uc].join(kt),modelMetadata:[ie,e,Wc].join(kt)}}function Pr(e){for(const t of Object.values(e))window.localStorage.removeItem(t)}function qc(e){const t=e.split(kt);if(t.length<3)throw new Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(kt)}function Hc(e){return e.startsWith(Xt.URL_SCHEME)?e.slice(Xt.URL_SCHEME.length):e}class Xt{constructor(t){if(!x().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,t==null||!t)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=t,this.keys=Br(this.modelPath)}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const n=JSON.stringify(t.modelTopology),s=JSON.stringify(t.weightSpecs),r=Pe(t);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,n),this.LS.setItem(this.keys.weightSpecs,s),this.LS.setItem(this.keys.weightData,Ic(t.weightData));const a={format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,signature:t.signature!=null?t.signature:void 0,userDefinedMetadata:t.userDefinedMetadata!=null?t.userDefinedMetadata:void 0,modelInitializer:t.modelInitializer!=null?t.modelInitializer:void 0,trainingConfig:t.trainingConfig!=null?t.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(a)),{modelArtifactsInfo:r}}catch{throw Pr(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){const t=JSON.parse(this.LS.getItem(this.keys.info));if(t==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(t.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const n={},s=JSON.parse(this.LS.getItem(this.keys.topology));if(s==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);n.modelTopology=s;const r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);n.weightSpecs=r;const a=this.LS.getItem(this.keys.modelMetadata);if(a!=null){const i=JSON.parse(a);n.format=i.format,n.generatedBy=i.generatedBy,n.convertedBy=i.convertedBy,i.signature!=null&&(n.signature=i.signature),i.userDefinedMetadata!=null&&(n.userDefinedMetadata=i.userDefinedMetadata),i.modelInitializer!=null&&(n.modelInitializer=i.modelInitializer),i.trainingConfig!=null&&(n.trainingConfig=i.trainingConfig)}const o=this.LS.getItem(this.keys.weightData);if(o==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return n.weightData=xc(o),n}}Xt.URL_SCHEME="localstorage://";const Lr=e=>x().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Xt.URL_SCHEME)?Gc(e.slice(Xt.URL_SCHEME.length)):null;K.registerSaveRouter(Lr);K.registerLoadRouter(Lr);function Gc(e){return new Xt(e)}class Mc{constructor(){d(x().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),d(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const t={},n=ie+kt,s=kt+Fr;for(let r=0;r<this.LS.length;++r){const a=this.LS.key(r);if(a.startsWith(n)&&a.endsWith(s)){const o=qc(a);t[o]=JSON.parse(this.LS.getItem(a))}}return t}async removeModel(t){t=Hc(t);const n=Br(t);if(this.LS.getItem(n.info)==null)throw new Error(`Cannot find model at path '${t}'`);const s=JSON.parse(this.LS.getItem(n.info));return Pr(n),s}}/**
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
 */const ue="://";class tt{constructor(){this.managers={}}static getInstance(){return tt.instance==null&&(tt.instance=new tt),tt.instance}static registerManager(t,n){d(t!=null,()=>"scheme must not be undefined or null."),t.endsWith(ue)&&(t=t.slice(0,t.indexOf(ue))),d(t.length>0,()=>"scheme must not be an empty string.");const s=tt.getInstance();d(s.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),s.managers[t]=n}static getManager(t){const n=tt.getInstance().managers[t];if(n==null)throw new Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(tt.getInstance().managers)}}function je(e){if(e.indexOf(ue)===-1)throw new Error(`The url string provided does not contain a scheme. Supported schemes are: ${tt.getSchemes().join(",")}`);return{scheme:e.split(ue)[0],path:e.split(ue)[1]}}async function Rr(e,t,n=!1){d(e!==t,()=>`Old path and new path are the same: '${e}'`);const s=K.getLoadHandlers(e);d(s.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),d(s.length<2,()=>`Copying failed because more than one (${s.length}) load handlers for source URL ${e}.`);const r=s[0],a=K.getSaveHandlers(t);d(a.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),d(a.length<2,()=>`Copying failed because more than one (${s.length}) save handlers for destination URL ${t}.`);const o=a[0],i=je(e).scheme,u=je(e).path,c=i===je(e).scheme,p=await r.load();n&&c&&await tt.getManager(i).removeModel(u);const h=await o.save(p);return n&&!c&&await tt.getManager(i).removeModel(u),h.modelArtifactsInfo}async function Yc(){const e=tt.getSchemes(),t={};for(const n of e){const s=await tt.getManager(n).listModels();for(const r in s){const a=n+ue+r;t[a]=s[r]}}return t}async function Xc(e){const t=je(e);return tt.getManager(t.scheme).removeModel(t.path)}async function Jc(e,t){return Rr(e,t,!1)}async function Zc(e,t){return Rr(e,t,!0)}/**
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
 */class Qc{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(t,n){return fetch(t,n)}now(){return performance.now()}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${n}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(t)}decode(t,n){return new TextDecoder(n).decode(t)}setTimeoutCustom(t,n){if(!window||!x().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(t,n);return}this.functionRefs.push(t),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},n),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",s=>{if(s.source===window&&s.data.name===this.messageName){s.stopPropagation();const r=this.functionRefs[s.data.index];r(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}}if(x().get("IS_BROWSER")){x().setPlatform("browser",new Qc);try{tt.registerManager(Xt.URL_SCHEME,new Mc)}catch{}try{tt.registerManager(Yt.URL_SCHEME,new Vc)}catch{}}/**
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
 */const tl={importFetch:()=>require("node-fetch")};let Tn;class el{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(t,n){return x().global.fetch!=null?x().global.fetch(t,n):(Tn==null&&(Tn=tl.importFetch()),Tn(t,n))}now(){const t=process.hrtime();return t[0]*1e3+t[1]/1e6}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${n}`);return this.textEncoder.encode(t)}decode(t,n){return t.length===0?"":new this.util.TextDecoder(n).decode(t)}}x().get("IS_NODE")&&!x().get("IS_BROWSER")&&x().setPlatform("node",new el);/**
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
 */function _t(e,t="float32",n){return t=t||"float32",rs(e),new xn(e,t,n)}/**
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
 */function nl(e,t){const n=f(e,"x","cast");if(!cr(t))throw new Error(`Failed to cast to unknown dtype ${t}`);if(t==="string"&&n.dtype!=="string"||t!=="string"&&n.dtype==="string")throw new Error("Only strings can be casted to strings");const s={x:n},r={dtype:t};return b.runKernel(yr,s,r)}const H=g({cast_:nl});/**
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
 */function sl(e){const n={x:f(e,"x","clone","string_or_numeric")};return b.runKernel(br,n)}const Ct=g({clone_:sl});/**
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
 */function zr(e,t=!1){console.log(e.toString(t))}/**
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
 */_r();const rl={buffer:_t,cast:H,clone:Ct,print:zr};dc(rl);/**
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
 */const al="model",ol=".json",il=".weights.bin";function Hs(e){return new Promise(t=>setTimeout(t)).then(e)}class Jt{constructor(t){if(!x().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");t.startsWith(Jt.URL_SCHEME)&&(t=t.slice(Jt.URL_SCHEME.length)),(t==null||t.length===0)&&(t=al),this.modelJsonFileName=t+ol,this.weightDataFileName=t+il}async save(t){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const n=window.URL.createObjectURL(new Blob([t.weightData],{type:"application/octet-stream"}));if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const s=[{paths:["./"+this.weightDataFileName],weights:t.weightSpecs}],r=Ar(t,s),a=window.URL.createObjectURL(new Blob([JSON.stringify(r)],{type:"application/json"})),o=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(o.download=this.modelJsonFileName,o.href=a,await Hs(()=>o.dispatchEvent(new MouseEvent("click"))),t.weightData!=null){const i=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;i.download=this.weightDataFileName,i.href=n,await Hs(()=>i.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:Pe(t)}}}}Jt.URL_SCHEME="downloads://";class ul{constructor(t){if(t==null||t.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${t}`);this.jsonFile=t[0],this.weightsFiles=t.slice(1)}async load(){return new Promise((t,n)=>{const s=new FileReader;s.onload=r=>{const a=JSON.parse(r.target.result),o=a.modelTopology;if(o==null){n(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(a.weightsManifest==null){n(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){t({modelTopology:o});return}const u=hs(a,c=>this.loadWeights(c));t(u)},s.onerror=r=>n(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),s.readAsText(this.jsonFile)})}loadWeights(t){const n=[],s=[];for(const o of t)n.push(...o.weights),s.push(...o.paths);const r=this.checkManifestAndWeightFiles(t),a=s.map(o=>this.loadWeightsFile(o,r[o]));return Promise.all(a).then(o=>[n,ls(o)])}loadWeightsFile(t,n){return new Promise((s,r)=>{const a=new FileReader;a.onload=o=>{const i=o.target.result;s(i)},a.onerror=o=>r(`Failed to weights data from file of path '${t}'.`),a.readAsArrayBuffer(n)})}checkManifestAndWeightFiles(t){const n=[],s=this.weightsFiles.map(a=>qs(a.name)),r={};for(const a of t)a.paths.forEach(o=>{const i=qs(o);if(n.indexOf(i)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${i}'`);if(n.push(i),s.indexOf(i)===-1)throw new Error(`Weight file with basename '${i}' is not provided.`);r[o]=this.weightsFiles[s.indexOf(i)]});if(n.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${n.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}const cl=e=>x().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Jt.URL_SCHEME)?ll(e.slice(Jt.URL_SCHEME.length)):null;K.registerSaveRouter(cl);function ll(e="model"){return new Jt(e)}function pl(e){return new ul(e)}/**
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
 */function Gs(e,t,n,s){o(e),n=n??0,s=s??1,i(n,s);let r=0;const a=u=>(u.then(c=>{const p=n+ ++r/e.length*(s-n);return t(p),c}),u);function o(u){d(u!=null&&Array.isArray(u)&&u.length>0,()=>"promises must be a none empty array")}function i(u,c){d(u>=0&&u<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${u}`),d(c>=0&&c<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${c}`),d(c>=u,()=>`startFraction must be no more than endFraction, but got startFraction ${u} and endFraction ${c}`)}return Promise.all(e.map(a))}/**
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
 */async function Vr(e,t){t==null&&(t={});const n=t.fetchFunc==null?x().platform.fetch:t.fetchFunc,s=e.map(h=>n(h,t.requestInit,{isBinary:!0})),r=0,a=.5,i=(t.onProgress==null?await Promise.all(s):await Gs(s,t.onProgress,r,a)).map(h=>h.arrayBuffer()),u=.5,c=1;return t.onProgress==null?await Promise.all(i):await Gs(i,t.onProgress,u,c)}async function hl(e,t="",n,s){return jr(o=>Vr(o,{requestInit:s}))(e,t,n)}function jr(e){return async(t,n="",s)=>{const r=t.map(()=>!1),a={},o=s!=null?s.map(()=>!1):[],i=[];if(t.forEach((y,T)=>{let N=0;y.weights.forEach(w=>{const k="quantization"in w?w.quantization.dtype:w.dtype,v=Bn[k]*q(w.shape),$=()=>{r[T]=!0,a[T]==null&&(a[T]=[]),a[T].push({manifestEntry:w,groupOffset:N,sizeBytes:v})};s!=null?s.forEach((E,I)=>{E===w.name&&($(),o[I]=!0)}):$(),i.push(w.name),N+=v})}),!o.every(y=>y)){const y=s.filter((T,N)=>!o[N]);throw new Error(`Could not find weights in manifest with names: ${y.join(", ")}. 
Manifest JSON has weights with names: ${i.join(", ")}.`)}const u=r.reduce((y,T,N)=>(T&&y.push(N),y),[]),c=[];u.forEach(y=>{t[y].paths.forEach(T=>{const N=n+(n.endsWith("/")?"":"/")+T;c.push(N)})});const p=await e(c),h={};let m=0;return u.forEach(y=>{const T=t[y].paths.length;let N=0;for(let E=0;E<T;E++)N+=p[m+E].byteLength;const w=new ArrayBuffer(N),k=new Uint8Array(w);let v=0;for(let E=0;E<T;E++){const I=new Uint8Array(p[m+E]);k.set(I,v),v+=I.byteLength}a[y].forEach(E=>{const I=w.slice(E.groupOffset,E.groupOffset+E.sizeBytes),_=Dr(I,[E.manifestEntry]);for(const A in _)h[A]=_[A]}),m+=T}),h}}/**
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
 */const fl="application/octet-stream",ml="application/json";class ms{constructor(t,n){if(this.DEFAULT_METHOD="POST",n==null&&(n={}),this.weightPathPrefix=n.weightPathPrefix,this.onProgress=n.onProgress,this.weightUrlConverter=n.weightUrlConverter,n.fetchFunc!=null?(d(typeof n.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=n.fetchFunc):this.fetch=x().platform.fetch,d(t!=null&&t.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(t)&&d(t.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${t.length}).`),this.path=t,n.requestInit!=null&&n.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=n.requestInit||{}}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const n=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);n.body=new FormData;const s=[{paths:["./model.weights.bin"],weights:t.weightSpecs}],r=Ar(t,s);n.body.append("model.json",new Blob([JSON.stringify(r)],{type:ml}),"model.json"),t.weightData!=null&&n.body.append("model.weights.bin",new Blob([t.weightData],{type:fl}),"model.weights.bin");const a=await this.fetch(this.path,n);if(a.ok)return{modelArtifactsInfo:Pe(t),responses:[a]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${a.status}.`)}async load(){const t=await this.fetch(this.path,this.requestInit);if(!t.ok)throw new Error(`Request to ${this.path} failed with status code ${t.status}. Please verify this URL points to the model JSON of the model to load.`);let n;try{n=await t.json()}catch{let o=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?o+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":o+=" Please make sure the server is serving valid JSON for this request.",new Error(o)}const s=n.modelTopology,r=n.weightsManifest;if(s==null&&r==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return hs(n,a=>this.loadWeights(a))}async loadWeights(t){const n=Array.isArray(this.path)?this.path[1]:this.path,[s,r]=dl(n),a=this.weightPathPrefix||s,o=fs(t),i=[],u=[];for(const p of t)for(const h of p.paths)this.weightUrlConverter!=null?u.push(this.weightUrlConverter(h)):i.push(a+h+r);this.weightUrlConverter&&i.push(...await Promise.all(u));const c=await Vr(i,{requestInit:this.requestInit,fetchFunc:this.fetch,onProgress:this.onProgress});return[o,ls(c)]}}ms.URL_SCHEME_REGEX=/^https?:\/\//;function dl(e){const t=e.lastIndexOf("/"),n=e.lastIndexOf("?"),s=e.substring(0,t),r=n>t?e.substring(n):"";return[s+"/",r]}function zn(e){return e.match(ms.URL_SCHEME_REGEX)!=null}const Kr=(e,t)=>{if(typeof fetch>"u"&&(t==null||t.fetchFunc==null))return null;{let n=!0;if(Array.isArray(e)?n=e.every(s=>zn(s)):n=zn(e),n)return ds(e,t)}return null};K.registerSaveRouter(Kr);K.registerLoadRouter(Kr);function ds(e,t){return new ms(e,t)}function gl(e,t){return ds(e,t)}/**
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
 */class Sn{constructor(t){this.modelArtifacts=t}load(){return this.modelArtifacts}}class Ur{constructor(t){this.saveHandler=t}save(t){return this.saveHandler(t)}}class yl{constructor(t){t.load&&(this.load=()=>Promise.resolve(t.load())),t.save&&(this.save=n=>Promise.resolve(t.save(n)))}}function bl(e,t,n,s){const r=arguments;return new yl(Ge(...r))}function Ge(e,t,n,s){return arguments.length===1?e.modelTopology!=null||e.weightSpecs!=null?new Sn(e):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Sn({modelTopology:e})):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Sn({modelTopology:e,weightSpecs:t,weightData:n,trainingConfig:s}))}function wl(e){return new Ur(e)}function Nl(e){return new Ur(e)}/**
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
 */const Wr=Object.freeze(Object.defineProperty({__proto__:null,browserFiles:pl,browserHTTPRequest:gl,concatenateArrayBuffers:ls,copyModel:Jc,decodeWeights:Dr,encodeWeights:_c,fromMemory:bl,fromMemorySync:Ge,getLoadHandlers:Lc,getModelArtifactsForJSON:hs,getModelArtifactsForJSONSync:ps,getModelArtifactsInfoForJSON:Pe,getSaveHandlers:Pc,getWeightSpecs:fs,http:ds,isHTTPScheme:zn,listModels:Yc,loadWeights:hl,moveModel:Zc,registerLoadRouter:Bc,registerSaveRouter:Fc,removeModel:Xc,weightsLoaderFactory:jr,withSaveHandler:wl,withSaveHandlerSync:Nl},Symbol.toStringTag,{value:"Module"}));/**
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
 */function Tl(e,t,n=!1,s=!1){let r=f(e,"a","matMul"),a=f(t,"b","matMul");[r,a]=U(r,a);const o={a:r,b:a},i={transposeA:n,transposeB:s};return b.runKernel(Io,o,i)}const P=g({matMul_:Tl});/**
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
 */function Sl(e,t,n=1,s=0,r="int32"){if(t<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${t}`);const o={indices:f(e,"indices","oneHot","int32")},i={dtype:r,depth:t,onValue:n,offValue:s};return b.runKernel(Ji,o,i)}const kl=g({oneHot_:Sl});/**
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
 */function uN(){x().set("PROD",!0)}function cN(){x().set("DEBUG",!0)}function lN(){x().set("DEPRECATION_WARNINGS_ENABLED",!1),console.warn("TensorFlow.js deprecation warnings have been disabled.")}function pN(e){x().getBool("DEPRECATION_WARNINGS_ENABLED")&&console.warn(e+" You can disable deprecation warnings with tf.disableDeprecationWarnings().")}function hN(){b.disposeVariables()}function fN(){return b}function mN(){return b.memory()}function dN(e){return b.profile(e)}function gt(e,t){return b.tidy(e,t)}function $l(e){us(e).forEach(n=>n.dispose())}function At(e){return b.keep(e)}function gN(e){return b.time(e)}function yN(e){return b.setBackend(e)}function bN(){return b.ready()}function wN(){return b.backendName}function NN(e){b.removeBackend(e)}function TN(e){return b.findBackend(e)}function SN(e){return b.findBackendFactory(e)}function kN(e,t,n=1){return b.registerBackend(e,t,n)}function $N(){return b.backend}function EN(e,t){x().setPlatform(e,t)}/**
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
 */function El(e){const n={input:f(e,"input","imag")};return b.runKernel(wi,n)}const an=g({imag_:El});/**
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
 */function _l(e){const n={x:f(e,"x","neg")};return b.runKernel(qi,n)}const $t=g({neg_:_l});/**
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
 */function vl(e){const n={input:f(e,"input","real")};return b.runKernel(ou,n)}const Ie=g({real_:vl});/**
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
 */function Il(e,t,n){const s=f(e,"x","transpose");if(t==null&&(t=s.shape.map((o,i)=>i).reverse()),d(s.rank===t.length,()=>`Error in transpose: rank of input ${s.rank} must match length of perm ${t}.`),t.forEach(o=>{d(o>=0&&o<s.rank,()=>`All entries in 'perm' must be between 0 and ${s.rank-1} but got ${t}`)}),s.rank<=1)return s.clone();const r={x:s},a={perm:t};return s.dtype==="complex64"?gt(()=>{let o=Ie(s),i=an(s);return o=b.runKernel(bn,{x:o},a),i=b.runKernel(bn,{x:i},a),n&&(i=$t(i)),Bt(o,i)}):b.runKernel(bn,r,a)}const Vn=g({transpose_:Il});/**
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
 */function xl(e,t){const n=e.length,s=[];for(let r=0;r<n;r++){const a=n-1-r,o=e[a]||1;(t[t.length-1-r]||1)>1&&o===1&&s.unshift(a)}return s}function qr(e,t){const n=[];for(let s=0;s<t.length;s++){const r=e[e.length-s-1],a=t.length-s-1,o=t[a];(r==null||r===1&&o>1)&&n.unshift(a)}return n}function G(e,t){const n=[],s=Math.max(e.length,t.length);for(let r=0;r<s;r++){let a=e[e.length-r-1];a==null&&(a=1);let o=t[t.length-r-1];if(o==null&&(o=1),a===1)n.unshift(o);else if(o===1)n.unshift(a);else if(a!==o){const i=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(i)}else n.unshift(a)}return n}const _N=Object.freeze(Object.defineProperty({__proto__:null,assertAndGetBroadcastShape:G,getBroadcastDims:xl,getReductionAxes:qr},Symbol.toStringTag,{value:"Module"}));/**
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
 */function Hr(e,t,n){if(ee(e),t!=null&&t.length!==3)throw new Error("tensor3d() requires shape to have three numbers");const s=Pt(e,n);if(s.length!==3&&s.length!==1)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return Lt(e,t,s,n)}/**
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
 */let zt;function Gr(e,t=3){if(t>4)throw new Error("Cannot construct Tensor with more than 4 channels from pixels.");if(e==null)throw new Error("pixels passed to tf.browser.fromPixels() can not be null");let n=!1,s=!1,r=!1,a=!1,o=!1,i=!1;if(e.data instanceof Uint8Array)n=!0;else if(typeof ImageData<"u"&&e instanceof ImageData)s=!0;else if(typeof HTMLVideoElement<"u"&&e instanceof HTMLVideoElement)r=!0;else if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement)a=!0;else if(e.getContext!=null)o=!0;else if(typeof ImageBitmap<"u"&&e instanceof ImageBitmap)i=!0;else throw new Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${e.constructor.name}`);if(En(Fs,b.backendName)!=null){const T={pixels:e},N={numChannels:t};return b.runKernel(Fs,T,N)}const[c,p]=r?[e.videoWidth,e.videoHeight]:[e.width,e.height];let h;if(o)h=e.getContext("2d").getImageData(0,0,c,p).data;else if(s||n)h=e.data;else if(a||r||i){if(zt==null)if(typeof document>"u")if(typeof OffscreenCanvas<"u"&&typeof OffscreenCanvasRenderingContext2D<"u")zt=new OffscreenCanvas(1,1).getContext("2d");else throw new Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.");else zt=document.createElement("canvas").getContext("2d",{willReadFrequently:!0});zt.canvas.width=c,zt.canvas.height=p,zt.drawImage(e,0,0,c,p),h=zt.getImageData(0,0,c,p).data}let m;if(t===4)m=new Int32Array(h);else{const T=c*p;m=new Int32Array(T*t);for(let N=0;N<T;N++)for(let w=0;w<t;++w)m[N*t+w]=h[N*4+w]}return Hr(m,[p,c,t],"int32")}function Dl(e){return e!=null&&e.data instanceof Uint8Array}function Al(){return typeof window<"u"&&typeof ImageBitmap<"u"&&window.hasOwnProperty("createImageBitmap")}function Ol(e){return e!=null&&e.width!==0&&e.height!==0}function Cl(e){return Al()&&!(e instanceof ImageBitmap)&&Ol(e)&&!Dl(e)}async function Fl(e,t=3){let n=null;if(x().getBool("WRAP_TO_IMAGEBITMAP")&&Cl(e)){let s;try{s=await createImageBitmap(e,{premultiplyAlpha:"none"})}catch{s=null}s!=null&&s.width===e.width&&s.height===e.height?n=s:n=e}else n=e;return Gr(n,t)}async function Bl(e,t){let n=f(e,"img","toPixels");if(!(e instanceof W)){const c=n;n=H(c,"int32"),c.dispose()}if(n.rank!==2&&n.rank!==3)throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${n.rank}.`);const[s,r]=n.shape.slice(0,2),a=n.rank===2?1:n.shape[2];if(a>4||a===2)throw new Error(`toPixels only supports depth of size 1, 3 or 4 but got ${a}`);if(n.dtype!=="float32"&&n.dtype!=="int32")throw new Error(`Unsupported type for toPixels: ${n.dtype}. Please use float32 or int32 tensors.`);const o=await n.data(),i=n.dtype==="float32"?255:1,u=new Uint8ClampedArray(r*s*4);for(let c=0;c<s*r;++c){const p=[0,0,0,255];for(let m=0;m<a;m++){const y=o[c*a+m];if(n.dtype==="float32"){if(y<0||y>1)throw new Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${y}.`)}else if(n.dtype==="int32"&&(y<0||y>255))throw new Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${y}.`);a===1?(p[0]=y*i,p[1]=y*i,p[2]=y*i):p[m]=y*i}const h=c*4;u[h+0]=Math.round(p[0]),u[h+1]=Math.round(p[1]),u[h+2]=Math.round(p[2]),u[h+3]=Math.round(p[3])}if(t!=null){t.width=r,t.height=s;const c=t.getContext("2d"),p=new ImageData(u,r,s);c.putImageData(p,0,0)}return n!==e&&n.dispose(),u}const Pl=g({fromPixels_:Gr}),vN=Object.freeze(Object.defineProperty({__proto__:null,fromPixels:Pl,fromPixelsAsync:Fl,toPixels:Bl},Symbol.toStringTag,{value:"Module"}));function Mr(e,t,n){const s=t.rank>1?t.shape[t.rank-1]:1,r=t.rank>1?t.rank-1:1,a=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${s}, and batchDim: ${r}.`;if(n.rank<r)throw new Error(a+` update.rank < ${r}. `);if(e.length<s+(n.rank-r))throw new Error(a+` Output shape length < ${s+(n.rank-r)}`);if(n.rank!==r+e.length-s)throw new Error(a+` update.rank != ${r+e.length-s}`);for(let o=0;o<r;++o)if(n.shape[o]!==t.shape[o])throw new Error(a+` updates.shape[${o}] (${n.shape[o]}) != indices.shape[${o}] (${t.shape[o]}).`);for(let o=0;o<n.rank-r;++o)if(n.shape[o+r]!==e[o+s])throw new Error(a+` updates.shape[${o+r}] (${n.shape[o+r]}) != shape[${o+r}] (${e[o+r]})`)}function Yr(e,t,n){if(t.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.dtype!=="int32")throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(n.length===0){if(t.size===0)throw new Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(e.size===0)throw new Error(`Updates specified for empty output. updates shape: ${e.shape}`)}Mr(n,t,e)}function Ll(e,t,n){const s=t.shape.length,r=s>1?t.shape[s-1]:1,a=n.length;let o=1;for(let h=r;h<a;++h)o*=n[h];const i=r<1?1:r,u=q(t.shape)/i,c=[...Fe(n.slice(0,r)),1],p=q(n);return{sliceRank:r,numUpdates:u,sliceSize:o,strides:c,outputSize:p}}const IN=Object.freeze(Object.defineProperty({__proto__:null,calculateShapes:Ll,validateInput:Yr,validateUpdateShape:Mr},Symbol.toStringTag,{value:"Module"}));/**
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
 */function Rl(e,t){let n=f(e,"a","add"),s=f(t,"b","add");[n,s]=U(n,s);const r={a:n,b:s};return b.runKernel(gr,r)}const Y=g({add_:Rl});/**
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
 */function zl(e,t){let n=f(e,"a","floorDiv"),s=f(t,"b","floorDiv");[n,s]=U(n,s);const r={a:n,b:s};return b.runKernel(hi,r)}const Xr=g({floorDiv_:zl});/**
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
 */function Vl(e,t){let n=f(e,"a","div"),s=f(t,"b","div");if([n,s]=U(n,s),n.dtype==="int32"&&s.dtype==="int32")return Xr(n,s);const r={a:n,b:s},a={};return b.runKernel(ti,r,a)}const st=g({div_:Vl});/**
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
 */function jl(e,t){let n=f(e,"a","mul"),s=f(t,"b","mul");[n,s]=U(n,s);const r={a:n,b:s};return b.runKernel(Wi,r)}const D=g({mul_:jl});/**
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
 */function Kl(e){const t=f(e,"x","abs");if(t.dtype==="complex64"){const n={x:t};return b.runKernel(Bo,n)}else{const n={x:t};return b.runKernel(ho,n)}}const ht=g({abs_:Kl});/**
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
 */function Ul(e){const n={x:f(e,"x","acos")};return b.runKernel(fo,n)}const Wl=g({acos_:Ul});/**
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
 */function ql(e){const n={x:f(e,"x","acosh")};return b.runKernel(mo,n)}const Hl=g({acosh_:ql});/**
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
 */function Gl(e){d(Array.isArray(e),()=>"The argument passed to tf.addN() must be a list of tensors"),d(e.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${e.length}`);const t=e.map((r,a)=>f(r,`tensors${a}`,"addN")),n=t[0];t.forEach(r=>{if(r.dtype!==n.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),t.forEach(r=>{if(!Et(r.shape,n.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const s=t;return b.runKernel(go,s)}const Ml=g({addN_:Gl});/**
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
 */function Yl(e,t=null,n=!1){const r={x:f(e,"x","all","bool")},a={axis:t,keepDims:n};return b.runKernel(yo,r,a)}const Xl=g({all_:Yl});/**
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
 */function Jl(e,t=null,n=!1){const r={x:f(e,"x","any","bool")},a={axis:t,keepDims:n};return b.runKernel(bo,r,a)}const Zl=g({any_:Jl});/**
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
 */function Ql(e,t=0){const s={x:f(e,"x","argMax")},r={axis:t};return b.runKernel(wo,s,r)}const tp=g({argMax_:Ql});/**
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
 */function ep(e,t=0){const s={x:f(e,"x","argMin")},r={axis:t};return b.runKernel(No,s,r)}const np=g({argMin_:ep});/**
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
 */function sp(e){const n={x:f(e,"x","asin")};return b.runKernel(To,n)}const rp=g({asin_:sp});/**
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
 */function ap(e){const n={x:f(e,"x","asinh")};return b.runKernel(So,n)}const op=g({asinh_:ap});/**
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
 */function ip(e){const n={x:f(e,"x","atan")};return b.runKernel(ko,n)}const up=g({atan_:ip});/**
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
 */function cp(e,t){let n=f(e,"a","atan2"),s=f(t,"b","atan2");[n,s]=U(n,s);const r={a:n,b:s};return b.runKernel(Eo,r)}const lp=g({atan2_:cp});/**
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
 */function pp(e){const n={x:f(e,"x","atanh")};return b.runKernel($o,n)}const hp=g({atanh_:pp});/**
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
 */function xN(e,t,n,s,r="NHWC",a){const o=e[3],i=[...t,o],u=wp(r);return on(e,i,n,a,s,null,null,u)}function fp(e,t,n,s,r,a,o="channelsLast"){const[i,u]=Me(t);let c;if(o==="channelsLast")c=[i,u,e[3],e[3]];else if(o==="channelsFirst")c=[i,u,e[1],e[1]];else throw new Error(`Unknown dataFormat ${o}`);return on(e,c,n,s,r,a,!1,o)}function DN(e,t,n,s,r,a,o="NDHWC"){const[i,u,c]=jn(t);let p,h;if(o==="NDHWC")h="channelsLast",p=[i,u,c,e[4],e[4]];else if(o==="NCDHW")h="channelsFirst",p=[i,u,c,e[1],e[1]];else throw new Error(`Unknown dataFormat ${o}`);return mp(e,p,n,s,r,!1,h,a)}function on(e,t,n,s,r,a,o=!1,i="channelsLast"){let[u,c,p,h]=[-1,-1,-1,-1];if(i==="channelsLast")[u,c,p,h]=e;else if(i==="channelsFirst")[u,h,c,p]=e;else throw new Error(`Unknown dataFormat ${i}`);const[m,y,,T]=t,[N,w]=Me(n),[k,v]=Me(s),$=ce(m,k),E=ce(y,v),{padInfo:I,outHeight:_,outWidth:A}=yp(r,c,p,N,w,$,E,a,i),F=o?T*h:T;let C;return i==="channelsFirst"?C=[u,F,_,A]:i==="channelsLast"&&(C=[u,_,A,F]),{batchSize:u,dataFormat:i,inHeight:c,inWidth:p,inChannels:h,outHeight:_,outWidth:A,outChannels:F,padInfo:I,strideHeight:N,strideWidth:w,filterHeight:m,filterWidth:y,effectiveFilterHeight:$,effectiveFilterWidth:E,dilationHeight:k,dilationWidth:v,inShape:e,outShape:C,filterShape:t}}function mp(e,t,n,s,r,a=!1,o="channelsLast",i){let[u,c,p,h,m]=[-1,-1,-1,-1,-1];if(o==="channelsLast")[u,c,p,h,m]=e;else if(o==="channelsFirst")[u,m,c,p,h]=e;else throw new Error(`Unknown dataFormat ${o}`);const[y,T,N,,w]=t,[k,v,$]=jn(n),[E,I,_]=jn(s),A=ce(y,E),F=ce(T,I),C=ce(N,_),{padInfo:z,outDepth:V,outHeight:Z,outWidth:X}=bp(r,c,p,h,k,v,$,A,F,C,i),Q=a?w*m:w;let it;return o==="channelsFirst"?it=[u,Q,V,Z,X]:o==="channelsLast"&&(it=[u,V,Z,X,Q]),{batchSize:u,dataFormat:o,inDepth:c,inHeight:p,inWidth:h,inChannels:m,outDepth:V,outHeight:Z,outWidth:X,outChannels:Q,padInfo:z,strideDepth:k,strideHeight:v,strideWidth:$,filterDepth:y,filterHeight:T,filterWidth:N,effectiveFilterDepth:A,effectiveFilterHeight:F,effectiveFilterWidth:C,dilationDepth:E,dilationHeight:I,dilationWidth:_,inShape:e,outShape:it,filterShape:t}}function dp(e,t,n,s,r){s==null&&(s=Jr(e,t,n));const a=e[0],o=e[1],i=Gt((a-t+2*s)/n+1,r),u=Gt((o-t+2*s)/n+1,r);return[i,u]}function gp(e,t,n,s,r,a){r==null&&(r=Jr(e,t,s));const o=e[0],i=e[1],u=e[2],c=Gt((o-t+2*r)/s+1,a),p=Gt((i-t+2*r)/s+1,a),h=Gt((u-t+2*r)/s+1,a);return[c,p,h,n]}function Jr(e,t,n,s=1){const r=ce(t,s);return Math.floor((e[0]*(n-1)-n+r)/2)}function Me(e){return typeof e=="number"?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function jn(e){return typeof e=="number"?[e,e,e]:e}function ce(e,t){return t<=1?e:e+(e-1)*(t-1)}function yp(e,t,n,s,r,a,o,i,u){let c,p,h;if(typeof e=="number"){c={top:e,bottom:e,left:e,right:e,type:e===0?"VALID":"NUMBER"};const y=dp([t,n],a,s,e,i);p=y[0],h=y[1]}else if(e==="same"){p=Math.ceil(t/s),h=Math.ceil(n/r);const m=Math.max(0,(p-1)*s+a-t),y=Math.max(0,(h-1)*r+o-n),T=Math.floor(m/2),N=m-T,w=Math.floor(y/2),k=y-w;c={top:T,bottom:N,left:w,right:k,type:"SAME"}}else if(e==="valid")c={top:0,bottom:0,left:0,right:0,type:"VALID"},p=Math.ceil((t-a+1)/s),h=Math.ceil((n-o+1)/r);else if(typeof e=="object"){const m=u==="channelsLast"?e[1][0]:e[2][0],y=u==="channelsLast"?e[1][1]:e[2][1],T=u==="channelsLast"?e[2][0]:e[3][0],N=u==="channelsLast"?e[2][1]:e[3][1];c={top:m,bottom:y,left:T,right:N,type:m===0&&y===0&&T===0&&N===0?"VALID":"EXPLICIT"},p=Gt((t-a+m+y)/s+1,i),h=Gt((n-o+T+N)/r+1,i)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:c,outHeight:p,outWidth:h}}function bp(e,t,n,s,r,a,o,i,u,c,p){let h,m,y,T;if(typeof e=="number"){h={top:e,bottom:e,left:e,right:e,front:e,back:e,type:e===0?"VALID":"NUMBER"};const w=gp([t,n,s,1],i,1,r,e,p);m=w[0],y=w[1],T=w[2]}else if(e==="same"){m=Math.ceil(t/r),y=Math.ceil(n/a),T=Math.ceil(s/o);const N=(m-1)*r+i-t,w=(y-1)*a+u-n,k=(T-1)*o+c-s,v=Math.floor(N/2),$=N-v,E=Math.floor(w/2),I=w-E,_=Math.floor(k/2),A=k-_;h={top:E,bottom:I,left:_,right:A,front:v,back:$,type:"SAME"}}else if(e==="valid")h={top:0,bottom:0,left:0,right:0,front:0,back:0,type:"VALID"},m=Math.ceil((t-i+1)/r),y=Math.ceil((n-u+1)/a),T=Math.ceil((s-c+1)/o);else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:h,outDepth:m,outHeight:y,outWidth:T}}function Gt(e,t){if(!t)return Math.trunc(e);switch(t){case"round":return Math.round(e);case"ceil":return Math.ceil(e);case"floor":return Math.floor(e);default:throw new Error(`Unknown roundingMode ${t}`)}}function Ye(e){const[t,n,s]=Me(e);return t===1&&n===1&&s===1}function Rt(e,t){return Ye(e)||Ye(t)}function wp(e){if(e==="NHWC")return"channelsLast";if(e==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${e}`)}function bt(e,t,n){if(n!=null){if(typeof t=="string")throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t=="number")d(he(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t=="object")t.forEach(s=>{s.forEach(r=>{d(he(r),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${r}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}/**
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
 */function Np(e,t){const s={x:f(e,"x","reshape","string_or_numeric")},r={shape:t};return b.runKernel(cu,s,r)}const S=g({reshape_:Np});/**
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
 */function Tp(e,t,n,s,r){const a=f(e,"x","avgPool","float32"),o=1;d(Rt(n,o),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`);let i=a,u=!1;a.rank===3&&(u=!0,i=S(a,[1,a.shape[0],a.shape[1],a.shape[2]])),d(i.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${i.rank}.`),bt("avgPool",s,r);const c={x:i},p={filterSize:t,strides:n,pad:s,dimRoundingMode:r};let h=b.runKernel(_o,c,p);return h=H(h,a.dtype),u?S(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const Zr=g({avgPool_:Tp});/**
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
 */function Sp(e,t,n,s,r,a="NDHWC"){const o=f(e,"x","avgPool3d","float32");let i=o,u=!1;o.rank===4&&(u=!0,i=S(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),d(i.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${i.rank}.`),d(a==="NDHWC",()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),bt("avgPool3d",s,r);const c={x:i},p={filterSize:t,strides:n,pad:s,dimRoundingMode:r,dataFormat:a};let h=b.runKernel(vo,c,p);return h=H(h,i.dtype),u?S(h,[h.shape[1],h.shape[2],h.shape[3],h.shape[4]]):h}const kp=g({avgPool3d_:Sp});/**
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
 */function $p(e,t=0){d(e.length>=1,()=>"Pass at least one tensor to concat");const n=ve(e,"tensors","concat","string_or_numeric");if(n[0].dtype==="complex64"&&n.forEach(a=>{if(a.dtype!=="complex64")throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${a.dtype}. `)}),n.length===1)return Ct(n[0]);const s=n,r={axis:t};return b.runKernel(Po,s,r)}const rt=g({concat_:$p});/**
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
 */function Ep(e){const n={x:f(e,"x","sigmoid","float32")};return b.runKernel($u,n)}const le=g({sigmoid_:Ep});/**
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
 */function _p(e,t,n){const s=f(e,"x","slice","string_or_numeric");if(s.rank===0)throw new Error("Slicing scalar is not possible");const r={x:s},a={begin:t,size:n};return b.runKernel(Nu,r,a)}const R=g({slice_:_p});/**
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
 */function vp(e){const n={x:f(e,"x","tanh","float32")};return b.runKernel(Uu,n)}const Kn=g({tanh_:vp});/**
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
 */function Ip(e,t,n,s,r,a){const o=f(e,"forgetBias","basicLSTMCell"),i=f(t,"lstmKernel","basicLSTMCell"),u=f(n,"lstmBias","basicLSTMCell"),c=f(s,"data","basicLSTMCell"),p=f(r,"c","basicLSTMCell"),h=f(a,"h","basicLSTMCell"),m=rt([c,h],1),y=P(m,i),T=Y(y,u),N=T.shape[0],w=T.shape[1]/4,k=[N,w],v=R(T,[0,0],k),$=R(T,[0,w],k),E=R(T,[0,w*2],k),I=R(T,[0,w*3],k),_=Y(D(le(v),Kn($)),D(p,le(Y(o,E)))),A=D(Kn(_),le(I));return[_,A]}const xp=g({basicLSTMCell_:Ip});/**
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
 */function Dp(e,t,n){const s=f(e,"x","batchToSpaceND"),r=t.reduce((i,u)=>i*u);d(s.rank>=1+t.length,()=>`input rank is ${s.rank} but should be > than blockShape.length ${t.length}`),d(n.length===t.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${t.length}`),d(s.shape[0]%r===0,()=>`input tensor batch is ${s.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(" * ")} === ${r}`);const a={x:s},o={blockShape:t,crops:n};return b.runKernel(xo,a,o)}const Qr=g({batchToSpaceND_:Dp});function Ap(e){let t;return e.rank===0||e.rank===1?t=S(e,[1,1,1,e.size]):e.rank===2?t=S(e,[1,1,e.shape[0],e.shape[1]]):e.rank===3?t=S(e,[1,e.shape[0],e.shape[1],e.shape[2]]):t=e,t}/**
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
 */function Op(e,t,n,s,r,a){a==null&&(a=.001);const o=f(e,"x","batchNorm"),i=f(t,"mean","batchNorm"),u=f(n,"variance","batchNorm");let c;r!=null&&(c=f(r,"scale","batchNorm"));let p;s!=null&&(p=f(s,"offset","batchNorm")),d(i.rank===u.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),d(p==null||i.rank===p.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),d(c==null||i.rank===c.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const m={x:Ap(o),scale:c,offset:p,mean:i,variance:u},y={varianceEpsilon:a},T=b.runKernel(fi,m,y);return S(T,o.shape)}const un=g({batchNorm_:Op});function Cp(e,t,n,s,r,a){const o=f(e,"x","batchNorm"),i=f(t,"mean","batchNorm"),u=f(n,"variance","batchNorm");let c;r!=null&&(c=f(r,"scale","batchNorm"));let p;return s!=null&&(p=f(s,"offset","batchNorm")),d(o.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${o.rank}.`),d(i.rank===2||i.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${i.rank}.`),d(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${u.rank}.`),c!=null&&d(c.rank===2||c.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${c.rank}.`),p!=null&&d(p.rank===2||p.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${p.rank}.`),un(o,i,u,p,c,a)}const Fp=g({batchNorm2d_:Cp});function Bp(e,t,n,s,r,a){const o=f(e,"x","batchNorm"),i=f(t,"mean","batchNorm"),u=f(n,"variance","batchNorm");let c;r!=null&&(c=f(r,"scale","batchNorm"));let p;return s!=null&&(p=f(s,"offset","batchNorm")),d(o.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${o.rank}.`),d(i.rank===3||i.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${i.rank}.`),d(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${u.rank}.`),c!=null&&d(c.rank===3||c.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${c.rank}.`),p!=null&&d(p.rank===3||p.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${p.rank}.`),un(o,i,u,p,c,a)}const Pp=g({batchNorm3d_:Bp});function Lp(e,t,n,s,r,a){const o=f(e,"x","batchNorm"),i=f(t,"mean","batchNorm"),u=f(n,"variance","batchNorm");let c;r!=null&&(c=f(r,"scale","batchNorm"));let p;return s!=null&&(p=f(s,"offset","batchNorm")),d(o.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${o.rank}.`),d(i.rank===4||i.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${i.rank}.`),d(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${u.rank}.`),c!=null&&d(c.rank===4||c.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${c.rank}.`),p!=null&&d(p.rank===4||p.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${p.rank}.`),un(o,i,u,p,c,a)}const Rp=g({batchNorm4d_:Lp});/**
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
 */function zp(e,t,n){const s=f(e,"x","bincount"),r=f(t,"weights","bincount");d(s.dtype==="int32",()=>`Error in bincount: input dtype must be int32, but got ${s.dtype}`),d(n>=0,()=>`size must be non-negative, but got ${n}.`),d(r.size===s.size||r.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${s.shape}, weights shape: ${r.shape}.`);const a={x:s,weights:r},o={size:n};return b.runKernel(Do,a,o)}const ta=g({bincount_:zp});/**
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
 */function Vp(e,t){const n=f(e,"s0","broadcastArgs","int32"),s=f(t,"s1","broadcastArgs","int32");if(n.rank!==1)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${n.rank}`);if(s.rank!==1)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${s.rank}`);const r={s0:n,s1:s};return b.runKernel(Ao,r)}const jp=g({broadcastArgs_:Vp});/**
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
 */function Kp(e,t){let n=f(e,"broadcastTo","x");const s=n.shape;if(t.some(c=>!(c>0)||c%1!==0))throw new Error(`broadcastTo(): Invalid broadcast shape [${t}].`);if(t.length<n.rank)throw new Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){const c=n.shape.slice();for(;c.length<t.length;)c.unshift(1);n=S(n,c)}const r=n.shape,a=Array.from(t);for(let c=t.length-1;c>=0;c--)if(r[c]===t[c])a[c]=1;else if(n.shape[c]!==1)throw new Error(`broadcastTo(): [${s}] cannot be broadcast to [${t}].`);if(a.map((c,p)=>c>1?p:-1).filter(c=>c>=0).length===0)return Ct(n);const i={x:n},u={reps:a};return b.runKernel(wr,i,u)}const Ke=g({broadcastTo_:Kp});/**
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
 */function Up(e){const n={x:f(e,"x","ceil","float32")};return b.runKernel(Oo,n)}const Wp=g({ceil_:Up});/**
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
 */function cn(e,t,n){const s={shape:e,value:t,dtype:n};return b.runKernel(ci,{},s)}/**
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
 */function qp(e,t,n){const s=f(e,"x","clipByValue");if(d(t<=n,()=>`Error in clip: min (${t}) must be less than or equal to max (${n}).`),t===n)return cn(s.shape,t,s.dtype);const r={x:s},a={clipValueMin:t,clipValueMax:n};return b.runKernel(Co,r,a)}const Hp=g({clipByValue_:qp});function Gp(e){return rt(e,0)}const Mp=g({concat1d_:Gp});function Yp(e,t){return rt(e,t)}const Xp=g({concat2d_:Yp});function Jp(e,t){return rt(e,t)}const Zp=g({concat3d_:Jp});function Qp(e,t){return rt(e,t)}const th=g({concat4d_:Qp});/**
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
 */function eh(e,t,n,s,r="NHWC",a=[1,1],o){const i=f(e,"x","conv2d","float32"),u=f(t,"filter","conv2d","float32");let c=i,p=!1;i.rank===3&&(p=!0,c=S(i,[1,i.shape[0],i.shape[1],i.shape[2]])),d(c.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${c.rank}.`),d(u.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${u.rank}.`),bt("conv2d",s,o);const h=r==="NHWC"?c.shape[3]:c.shape[1];d(h===u.shape[2],()=>`Error in conv2d: depth of input (${h}) must match input depth for filter ${u.shape[2]}.`),d(Rt(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);const m={x:c,filter:u},y={strides:n,pad:s,dataFormat:r,dilations:a,dimRoundingMode:o},T=b.runKernel(Lo,m,y);return p?S(T,[T.shape[1],T.shape[2],T.shape[3]]):T}const ln=g({conv2d_:eh});function nh(e,t,n,s,r="NWC",a=1,o){const i=f(e,"x","conv1d"),u=f(t,"filter","conv1d");let c=i,p=!1;i.rank===2&&(p=!0,c=S(i,[1,i.shape[0],i.shape[1]])),d(c.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${c.rank}.`),d(u.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${u.rank}.`),bt("conv1d",s,o),d(c.shape[2]===u.shape[1],()=>`Error in conv1d: depth of input (${c.shape[2]}) must match input depth for filter ${u.shape[1]}.`),d(Rt(n,a),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${a}'`),d(r==="NWC",()=>`Error in conv1d: got dataFormat of ${r} but only NWC is currently supported.`);const h=S(u,[1,u.shape[0],u.shape[1],u.shape[2]]),m=S(c,[c.shape[0],1,c.shape[1],c.shape[2]]),w=ln(m,h,[1,n],s,"NHWC",[1,a],o);return p?S(w,[w.shape[2],w.shape[3]]):S(w,[w.shape[0],w.shape[2],w.shape[3]])}const sh=g({conv1d_:nh});/**
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
 */function rh(e,t,n,s,r,a="NHWC",o){d(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let i=e,u=t,c=!1;t.rank===3&&(c=!0,u=S(t,[1,t.shape[0],t.shape[1],t.shape[2]]),i=[1,e[0],e[1],e[2]]),d(i.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${i.length}.`),d(u.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${u.rank}`),d(n.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);const p=a==="NHWC"?i[3]:i[1],h=a==="NHWC"?u.shape[3]:u.shape[1];d(p===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${p}) must match input depth for filter ${n.shape[2]}.`),d(h===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${h}) must match output depth for filter ${n.shape[3]}.`),bt("conv2dDerInput",r,o);const m={dy:u,filter:n},y={strides:s,pad:r,dataFormat:a,dimRoundingMode:o,inputShape:i},T=b.runKernel(zo,m,y);return c?S(T,[T.shape[1],T.shape[2],T.shape[3]]):T}const ea=g({conv2DBackpropInput_:rh});function ah(e,t,n,s,r,a){const o=f(e,"x","conv2dTranspose"),i=f(t,"filter","conv2dTranspose");return ea(n,o,i,s,r,"NHWC",a)}const oh=g({conv2dTranspose_:ah});/**
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
 */function ih(e,t,n,s,r="NDHWC",a=[1,1,1]){const o=f(e,"x","conv3d"),i=f(t,"filter","conv3d");let u=o,c=!1;o.rank===4&&(c=!0,u=S(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),d(u.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${u.rank}.`),d(i.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${i.rank}.`),d(u.shape[4]===i.shape[3],()=>`Error in conv3d: depth of input (${u.shape[4]}) must match input depth for filter ${i.shape[3]}.`),d(Rt(n,a),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),d(r==="NDHWC",()=>`Error in conv3d: got dataFormat of ${r} but only NDHWC is currently supported.`);const p={x:u,filter:i},h={strides:n,pad:s,dataFormat:r,dilations:a},m=b.runKernel(Vo,p,h);return c?S(m,[m.shape[1],m.shape[2],m.shape[3],m.shape[4]]):m}const uh=g({conv3d_:ih});/**
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
 */function ch(e,t,n,s,r){d(e.length===t.rank,()=>`Length of inShape (${e.length}) and rank of dy (${t.rank}) must match`);let a=e,o=t,i=!1;t.rank===4&&(i=!0,o=S(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),a=[1,e[0],e[1],e[2],e[3]]);const u=a[4],c=o.shape[4];d(a.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${a.length}.`),d(o.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${o.rank}`),d(n.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),d(u===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${u}) must match input depth for filter ${n.shape[3]}.`),d(c===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${c}) must match output depth for filter ${n.shape[4]}.`);const p={dy:o,filter:n},h={pad:r,strides:s,inputShape:a},m=b.runKernel(jo,p,h);return i?S(m,[m.shape[1],m.shape[2],m.shape[3],m.shape[4]]):m}const lh=g({conv3DBackpropInput_:ch});function ph(e,t,n,s,r){const a=f(e,"x","conv3dTranspose"),o=f(t,"filter","conv3dTranspose");return lh(n,a,o,s,r)}const hh=g({conv3dTranspose_:ph});/**
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
 */function fh(e){const n={x:f(e,"x","cos","float32")};return b.runKernel(Ko,n)}const mh=g({cos_:fh});/**
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
 */function dh(e){const n={x:f(e,"x","cosh","float32")};return b.runKernel(Uo,n)}const gh=g({cosh_:dh});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
 */function yh(e,t=0,n=!1,s=!1){const a={x:f(e,"x","cumprod")},o={axis:t,exclusive:n,reverse:s};return b.runKernel(Wo,a,o)}const bh=g({cumprod_:yh});/**
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
 */function wh(e,t=0,n=!1,s=!1){const a={x:f(e,"x","cumsum")},o={axis:t,exclusive:n,reverse:s};return b.runKernel(qo,a,o)}const Nh=g({cumsum_:wh});/**
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
 */function Th(e,t,n,s=!1){const r=f(e,"x","denseBincount"),a=f(t,"weights","denseBincount");d(r.dtype==="int32",()=>`Error in denseBincount: input dtype must be int32, but got ${r.dtype}`),d(r.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${r.rank}.`),d(n>=0,()=>`size must be non-negative, but got ${n}.`),d(a.size===r.size||a.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${r.shape}, weights shape: ${a.shape}.`);const o={x:r,weights:a},i={size:n,binaryOutput:s};return b.runKernel(Go,o,i)}const Sh=g({denseBincount_:Th});/**
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
 */function kh(e,t,n="NHWC"){const s=f(e,"x","depthToSpace","float32"),r=n==="NHWC"?s.shape[1]:s.shape[2],a=n==="NHWC"?s.shape[2]:s.shape[3],o=n==="NHWC"?s.shape[3]:s.shape[1];d(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),d(r*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${r} and ${t}  for depthToSpace with input shape
    ${s.shape}`),d(a*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${a} and ${t} for depthToSpace with input shape
        ${s.shape}`),d(o%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${o} for depthToSpace with input shape ${s.shape}`);const i={x:s},u={blockSize:t,dataFormat:n};return b.runKernel(Mo,i,u)}const $h=g({depthToSpace_:kh});/**
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
 */function Eh(e,t,n,s,r="NHWC",a=[1,1],o){const i=f(e,"x","depthwiseConv2d","float32"),u=f(t,"filter","depthwiseConv2d","float32");let c=i,p=!1;i.rank===3&&(p=!0,c=S(i,[1,i.shape[0],i.shape[1],i.shape[2]])),d(c.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${c.rank}.`),d(u.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${u.rank}.`);const h=r==="NHWC"?c.shape[3]:c.shape[1];d(h===u.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${h}) must match the inChannels dimension in filter ${u.shape[2]}.`),bt("depthwiseConv2d",s,o);const m={x:c,filter:u},y={strides:n,pad:s,dataFormat:r,dilations:a,dimRoundingMode:o},T=b.runKernel(Yo,m,y);return p?S(T,[T.shape[1],T.shape[2],T.shape[3]]):T}const gs=g({depthwiseConv2d_:Eh});/**
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
 */function _h(e){const n={x:f(e,"x","diag")};return b.runKernel(Zo,n)}const vh=g({diag_:_h});/**
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
 */function Ih(e,t,n,s,r=[1,1],a="NHWC"){const o=f(e,"x","dilation2d"),i=f(t,"filter","dilation2d");d(o.rank===3||o.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${o.rank}.`),d(i.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${i.rank}.`),d(a==="NHWC",()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${a}`);let u=o,c=!1;o.rank===3&&(u=S(o,[1,o.shape[0],o.shape[1],o.shape[2]]),c=!0);const p={x:u,filter:i},h={strides:n,pad:s,dilations:r},m=b.runKernel(Qo,p,h);return c?S(m,[m.shape[1],m.shape[2],m.shape[3]]):m}const xh=g({dilation2d_:Ih});/**
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
 */function Dh(e,t){let n=f(e,"a","equal","string_or_numeric"),s=f(t,"b","equal","string_or_numeric");[n,s]=U(n,s),G(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(ri,r)}const na=g({equal_:Dh});/**
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
 */function Ah(e,t,n){const s=f(t,"a","where"),r=f(n,"b","where"),a=f(e,"condition","where","bool"),o=G(G(a.shape,s.shape),r.shape),i=Ke(a,o),u=Ke(s,o),c=Ke(r,o),p={condition:i,t:u,e:c};return b.runKernel(bu,p)}const de=g({where_:Ah});/**
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
 */function Oh(e){const n={x:f(e,"x","zerosLike")};return b.runKernel(Yu,n)}const ys=g({zerosLike_:Oh});/**
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
 */function Ch(e,t){let n=f(e,"a","div"),s=f(t,"b","div");[n,s]=U(n,s);const r=st(n,s),a=ys(r),o=na(s,a);return de(o,a,r)}const Fh=g({divNoNan_:Ch});/**
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
 */function Bh(e,t){const n=f(e,"t1","dot"),s=f(t,"t2","dot");d((n.rank===1||n.rank===2)&&(s.rank===1||s.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${s.rank}.`);const r=n.rank===1?n.size:n.shape[1],a=s.rank===1?s.size:s.shape[0];if(d(r===a,()=>`Error in dot: inner dimensions of inputs must match, but got ${r} and ${a}.`),n.rank===1&&s.rank===1){const o=S(n,[1,-1]),i=S(s,[-1,1]),u=P(o,i);return S(u,[])}else if(n.rank===1&&s.rank===2){const o=S(n,[1,-1]),i=S(s,[s.shape[0],s.shape[1]]),u=P(o,i);return S(u,[u.size])}else if(n.rank===2&&s.rank===1){const o=S(s,[-1,1]),i=P(n,o);return S(i,[i.size])}else{const o=S(s,[s.shape[0],s.shape[1]]);return P(n,o)}}const Ph=g({dot_:Bh});/**
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
 */function Lh(e,...t){const n=t.map((r,a)=>f(r,`tensors${a}`,"einsum")),s={equation:e};return b.runKernel(ei,n,s)}const Rh=g({einsum_:Lh});/**
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
 */function zh(e){const n={x:f(e,"x","elu","float32")};return b.runKernel(ni,n)}const sa=g({elu_:zh});/**
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
 */function Vh(e){let t=f(e,"x","erf");d(t.dtype==="int32"||t.dtype==="float32",()=>"Input dtype must be `int32` or `float32`."),t.dtype==="int32"&&(t=H(t,"float32"));const n={x:t};return b.runKernel(si,n)}const jh=g({erf_:Vh});/**
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
 */function ra(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function Kh(e,t,n){const s=e.length+t.length,r=[];let a=0,o=0;for(let i=0;i<s;i++)n.indexOf(i)===-1?r.push(e[a++]):r.push(t[o++]);return r}function AN(e,t){const n=[],s=e.length;for(let a=0;a<s;a++)t.indexOf(a)===-1&&n.push(e[a]);const r=t.map(a=>e[a]);return[n,r]}function pn(e,t){const n=t.map(s=>1);return Kh(e,n,t)}function ON(e,t,n){d(ra(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function CN(e,t){if(ra(e,t))return null;const n=[];for(let s=0;s<t;++s)e.indexOf(s)===-1&&n.push(s);return e.forEach(s=>n.push(s)),n}function FN(e){return e.map((t,n)=>[n,t]).sort((t,n)=>t[1]-n[1]).map(t=>t[0])}function BN(e,t){const n=[];for(let s=t-e;s<t;++s)n.push(s);return n}/**
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
 */function Uh(e,t=null,n=!1){const r={x:f(e,"x","max")},a={reductionIndices:t,keepDims:n};return b.runKernel(Ci,r,a)}const pe=g({max_:Uh});/**
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
 */function Wh(e,t=null,n=!1){const r={x:f(e,"x","min")},a={axis:t,keepDims:n};return b.runKernel(zi,r,a)}const Un=g({min_:Wh});/**
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
 */function qh(e,t){let n=f(e,"base","pow"),s=f(t,"exp","pow");[n,s]=U(n,s);const r={a:n,b:s};return b.runKernel(tu,r)}const bs=g({pow_:qh});/**
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
 */function B(e,t){if((yt(e)&&t!=="string"||Array.isArray(e))&&t!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(t==="string"&&yt(e)&&!(e instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return Lt(e,[],[],t)}/**
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
 */function Hh(e){const n={x:f(e,"x","sqrt","float32")};return b.runKernel(_u,n)}const Wn=g({sqrt_:Hh});/**
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
 */function Gh(e){const t=f(e,"x","square"),n={};return b.runKernel("Square",{x:t},n)}const hn=g({square_:Gh});/**
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
 */function Mh(e,t=null,n=!1){let s=f(e,"x","sum");s.dtype==="bool"&&(s=H(s,"int32"));const r={x:s},a={axis:t,keepDims:n};return b.runKernel(vu,r,a)}const j=g({sum_:Mh});/**
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
 */function Yh(e,t="euclidean",n=null,s=!1){e=f(e,"x","norm");const r=aa(e,t,n);let a=r.shape;if(s){const o=Ce(n,e.shape);a=pn(r.shape,o)}return S(r,a)}function aa(e,t,n=null){if(e.rank===0)return ht(e);if(e.rank!==1&&n===null)return aa(S(e,[-1]),t,n);if(e.rank===1||typeof n=="number"||Array.isArray(n)&&n.length===1){if(t===1)return j(ht(e),n);if(t===1/0)return pe(ht(e),n);if(t===-1/0)return Un(ht(e),n);if(t==="euclidean"||t===2)return Wn(j(bs(ht(e),B(2,"int32")),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(n)&&n.length===2){if(t===1)return pe(j(ht(e),n[0]),n[1]-1);if(t===1/0)return pe(j(ht(e),n[1]),n[0]);if(t===-1/0)return Un(j(ht(e),n[1]),n[0]);if(t==="fro"||t==="euclidean")return Wn(j(hn(e),n));throw new Error(`Error in norm: invalid ord value: ${t}`)}throw new Error(`Error in norm: invalid axis: ${n}`)}const fn=g({norm_:Yh});/**
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
 */function Xh(e,t=null,n=!1){return fn(e,"euclidean",t,n)}const Jh=g({euclideanNorm_:Xh});/**
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
 */function Zh(e){const n={x:f(e,"x","exp")};return b.runKernel(ai,n)}const Zt=g({exp_:Zh});/**
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
 */function Qh(e,t=0){const n=f(e,"x","expandDims","string_or_numeric");d(t<=n.rank,()=>"Axis must be <= rank of the tensor");const s={input:n},r={dim:t};return b.runKernel(oi,s,r)}const jt=g({expandDims_:Qh});/**
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
 */function tf(e){const n={x:f(e,"x","expm1")};return b.runKernel(ii,n)}const ef=g({expm1_:tf});/**
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
 */function nf(e,t){const n=f(e,"x","tile","string_or_numeric");d(n.rank===t.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${t}.`);const s={x:n},r={reps:t};return b.runKernel(wr,s,r)}const ke=g({tile_:nf});/**
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
 */function sf(e,t,n,s="float32"){t==null&&(t=e);const r=_t([e,t],s),a=e<=t?e:t;for(let i=0;i<a;++i)r.set(1,i,i);const o=S(r.toTensor(),[e,t]);if(n==null)return o;if(n.length===1)return ke(jt(o,0),[n[0],1,1]);if(n.length===2)return ke(jt(jt(o,0),0),[n[0],n[1],1,1]);if(n.length===3)return ke(jt(jt(jt(o,0),0),0),[n[0],n[1],n[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}const oa=g({eye_:sf});/**
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
 */function rf(e){const n={x:f(e,"x","floor","float32")};return b.runKernel(pi,n)}const ia=g({floor_:rf});/**
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
 */function af(e,t,n=0,s=0){const r=f(e,"x","gather"),a=f(t,"indices","gather","int32"),o={x:r,indices:a},i={axis:n,batchDims:s};return b.runKernel(mi,o,i)}const ua=g({gather_:af});/**
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
 */function of(e,t){let n=f(e,"a","greater","string_or_numeric"),s=f(t,"b","greater","string_or_numeric");[n,s]=U(n,s),G(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(gi,r)}const mn=g({greater_:of});/**
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
 */function uf(e,t){let n=f(e,"a","greaterEqual","string_or_numeric"),s=f(t,"b","greaterEqual","string_or_numeric");[n,s]=U(n,s),G(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(yi,r)}const ca=g({greaterEqual_:uf});/**
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
 */function cf(e){const n={x:f(e,"x","isFinite")};return b.runKernel(Ni,n)}const lf=g({isFinite_:cf});/**
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
 */function pf(e){const n={x:f(e,"x","isInf")};return b.runKernel(Ti,n)}const hf=g({isInf_:pf});/**
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
 */function ff(e){const n={x:f(e,"x","isNaN")};return b.runKernel(Si,n)}const mf=g({isNaN_:ff});/**
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
 */function df(e,t=.2){const s={x:f(e,"x","leakyRelu")},r={alpha:t};return b.runKernel(ki,s,r)}const la=g({leakyRelu_:df});/**
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
 */function gf(e,t){let n=f(e,"a","less","string_or_numeric"),s=f(t,"b","less","string_or_numeric");[n,s]=U(n,s),G(n.shape,s.shape);const r={a:n,b:s};return b.runKernel($i,r)}const yf=g({less_:gf});/**
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
 */function bf(e,t){let n=f(e,"a","lessEqual","string_or_numeric"),s=f(t,"b","lessEqual","string_or_numeric");[n,s]=U(n,s),G(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Ei,r)}const ws=g({lessEqual_:bf});/**
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
 */function wf(e,t,n){if(n<=0)throw new Error("The number of values should be positive.");const s={start:e,stop:t,num:n};return b.runKernel(_i,{},s)}/**
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
 */function Nf(e,t=5,n=1,s=1,r=.5){const a=f(e,"x","localResponseNormalization");d(a.rank===4||a.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${a.rank}.`),d(he(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let o=a,i=!1;a.rank===3&&(i=!0,o=S(a,[1,a.shape[0],a.shape[1],a.shape[2]]));const u={x:o},c={depthRadius:t,bias:n,alpha:s,beta:r},p=b.runKernel(Oi,u,c);return i?S(p,[p.shape[1],p.shape[2],p.shape[3]]):p}const Tf=g({localResponseNormalization_:Nf});/**
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
 */function Sf(e){const n={x:f(e,"x","log","float32")};return b.runKernel(vi,n)}const xe=g({log_:Sf});/**
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
 */function kf(e){const n={x:f(e,"x","log1p")};return b.runKernel(Ii,n)}const pa=g({log1p_:kf});/**
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
 */function PN(e){return d(Ft(e),()=>"The f passed in grad(f) must be a function"),(t,n)=>{const s=f(t,"x","tf.grad","string_or_numeric"),r=n!=null?f(n,"dy","tf.grad"):null;return b.tidy(()=>{const{value:a,grads:o}=b.gradients(()=>e(s),[s],r);return r!=null&&at(a.shape,r.shape,"The shape of dy passed in grad(f)(x, dy) must match the shape returned by f(x)"),dn(o),o[0]})}}function LN(e){return d(Ft(e),()=>"The f passed in grads(f) must be a function"),(t,n)=>{d(Array.isArray(t),()=>"The args passed in grads(f)(args) must be an array of `Tensor`s or `TensorLike`s");const s=ve(t,"args","tf.grads","string_or_numeric"),r=n!=null?f(n,"dy","tf.grads"):null;return b.tidy(()=>{const{value:a,grads:o}=b.gradients(()=>e(...s),s,r);return r!=null&&at(a.shape,r.shape,"The shape of dy passed in grads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),dn(o),o})}}function RN(e){return d(Ft(e),()=>"The f passed in valueAndGrad(f) must be a function"),(t,n)=>{d(t instanceof W,()=>"The x passed in valueAndGrad(f)(x) must be a tensor"),d(n==null||n instanceof W,()=>"The dy passed in valueAndGrad(f)(x, dy) must be a tensor");const{grads:s,value:r}=b.gradients(()=>e(t),[t],n);return dn(s),{grad:s[0],value:r}}}function zN(e){return d(Ft(e),()=>"The f passed in valueAndGrads(f) must be a function"),(t,n)=>{d(Array.isArray(t)&&t.every(r=>r instanceof W),()=>"The args passed in valueAndGrads(f)(args) must be array of tensors"),d(n==null||n instanceof W,()=>"The dy passed in valueAndGrads(f)(args, dy) must be a tensor");const s=b.gradients(()=>e(...t),t,n);return n!=null&&at(s.value.shape,n.shape,"The shape of dy passed in valueAndGrads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),dn(s.grads),s}}function VN(e,t){d(Ft(e),()=>"The f passed in variableGrads(f) must be a function"),d(t==null||Array.isArray(t)&&t.every(c=>c instanceof qe),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const n=t!=null;if(!n){t=[];for(const c in b.registeredVariables)t.push(b.registeredVariables[c])}const s=n?t.filter(c=>!c.trainable):null,r=t.length;t=t.filter(c=>c.trainable),d(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${r} variables is trainable.`);const a=!0,{value:o,grads:i}=b.gradients(e,t,null,a);d(i.some(c=>c!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),d(o.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${o.rank} tensor`);const u={};return t.forEach((c,p)=>{i[p]!=null&&(u[c.name]=i[p])}),s!=null&&s.forEach(c=>u[c.name]=null),{value:o,grads:u}}function vt(e){return b.customGrad(e)}function dn(e){if(e.filter(n=>n==null).length>0)throw new Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`)}/**
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
 */function $f(e){const n={x:f(e,"x","softplus")};return b.runKernel(Eu,n)}const ha=g({softplus_:$f});/**
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
 */function Ef(e){const t=f(e,"x","logSigmoid");return vt(s=>({value:$t(ha($t(s))),gradFunc:o=>D(o,le($t(s)))}))(t)}const _f=g({logSigmoid_:Ef});/**
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
 */function vf(e,t){let n=f(e,"a","sub"),s=f(t,"b","sub");[n,s]=U(n,s);const r={a:n,b:s};return b.runKernel(ju,r)}const O=g({sub_:vf});/**
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
 */function If(e,t=-1){const n=f(e,"logits","logSoftmax");if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${t}`);return vt((r,a)=>{const i=pe(r,t,!0),u=O(r,i),c=O(H(u,"float32"),xe(j(Zt(u),t,!0)));return a([c]),{value:c,gradFunc:(h,m)=>{const[y]=m,T=!0,N=Zt(y);return O(h,D(j(h,t,T),N))}}})(n)}const xf=g({logSoftmax_:If});/**
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
 */function Df(e,t=null,n=!1){const s=f(e,"x","logSumExp"),r=Ce(t,s.shape),a=pe(s,r,!0),o=O(s,a),i=Zt(o),u=j(i,r),c=xe(u),p=Y(S(a,c.shape),c);if(n){const h=pn(p.shape,r);return S(p,h)}return p}const fa=g({logSumExp_:Df});/**
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
 */function Af(e,t){const n=f(e,"a","logicalAnd","bool"),s=f(t,"b","logicalAnd","bool");G(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(xi,r)}const Xe=g({logicalAnd_:Af});/**
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
 */function Of(e){const n={x:f(e,"x","logicalNot","bool")};return b.runKernel(Di,n)}const ma=g({logicalNot_:Of});/**
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
 */function Cf(e,t){const n=f(e,"a","logicalOr","bool"),s=f(t,"b","logicalOr","bool");G(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Ai,r)}const da=g({logicalOr_:Cf});/**
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
 */function Ff(e,t){const n=f(e,"a","logicalXor","bool"),s=f(t,"b","logicalXor","bool");return G(n.shape,s.shape),Xe(da(e,t),ma(Xe(e,t)))}const Bf=g({logicalXor_:Ff});/**
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
 */const ze=2147483648;function Pf(e,t,n="left"){const s=f(e,"sortedSequence","searchSorted"),r=f(t,"values","searchSorted"),a=s.shape[s.shape.length-1],o=r.shape[r.shape.length-1],i=S(s,[-1,a]),u=S(r,[-1,o]);if(i.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(i.shape[0]!==u.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(q(u.shape)>=ze)throw new Error(`values tensor size must less than ${ze}`);if(i.shape[1]>=ze)throw new Error(`trailing dim_size must less than ${ze} for int32 output type, was ${i.shape[1]}`);const c={sortedSequence:i,values:u},p={side:n};return b.runKernel(yu,c,p)}const Ns=g({searchSorted_:Pf});/**
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
 */function Lf(e,t){return Ns(e,t,"left")}/**
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
 */function Rf(e,t,n,s,r){const a=f(e,"x","maxPool"),o=1;let i=a,u=!1;a.rank===3&&(u=!0,i=S(a,[1,a.shape[0],a.shape[1],a.shape[2]])),d(i.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${i.rank}.`),d(Rt(n,o),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`),bt("maxPool",s,r);const c={x:i},p={filterSize:t,strides:n,pad:s,dimRoundingMode:r},h=b.runKernel(Bi,c,p);return u?S(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const ga=g({maxPool_:Rf});/**
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
 */function zf(e,t=[1,1,1],n,s,r,a="NDHWC"){const o=f(e,"x","maxPool3d");let i=o,u=!1;o.rank===4&&(u=!0,i=S(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),d(i.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${i.rank}.`),d(a==="NDHWC",()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),bt("maxPool3d",s,r);const c={x:i},p={filterSize:t,strides:n,pad:s,dimRoundingMode:r,dataFormat:a},h=b.runKernel(Pi,c,p);return u?S(h,[h.shape[1],h.shape[2],h.shape[3],h.shape[4]]):h}const Vf=g({maxPool3d_:zf});/**
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
 */function jf(e,t,n,s,r=!1){const o={x:f(e,"x","maxPoolWithArgmax")},i={filterSize:t,strides:n,pad:s,includeBatchInIndex:r},u=b.runKernel(Li,o,i);return{result:u[0],indexes:u[1]}}const Kf=g({maxPoolWithArgmax_:jf});/**
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
 */function Uf(e,t){let n=f(e,"a","maximum"),s=f(t,"b","maximum");[n,s]=U(n,s),n.dtype==="bool"&&(n=H(n,"int32"),s=H(s,"int32")),G(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Fi,r)}const Wf=g({maximum_:Uf});/**
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
 */function qf(e,t=null,n=!1){const r={x:f(e,"x","mean")},a={axis:t,keepDims:n};return b.runKernel(Ri,r,a)}const Je=g({mean_:qf});/**
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
 */function ge(e,t="float32"){if(t==="complex64"){const s=ge(e,"float32"),r=ge(e,"float32");return Bt(s,r)}const n=sn(q(e),t);return b.makeTensor(n,e,t)}/**
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
 */function qt(e,t="float32"){if(t==="complex64"){const s=qt(e,"float32"),r=ge(e,"float32");return Bt(s,r)}const n=ss(q(e),t);return b.makeTensor(n,e,t)}/**
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
 */function Hf(e,t,{indexing:n="xy"}={}){if(n!=="xy"&&n!=="ij")throw new TypeError(`${n} is not a valid third argument to meshgrid`);if(e===void 0)return[];let s=f(e,"x","meshgrid",e instanceof W?e.dtype:"float32");if(t===void 0)return[s];let r=f(t,"y","meshgrid",t instanceof W?t.dtype:"float32");const a=q(s.shape),o=q(r.shape);return n==="xy"?(s=S(s,[1,-1]),r=S(r,[-1,1]),[P(qt([o,1],s.dtype),s),P(r,qt([1,a],r.dtype))]):(s=S(s,[-1,1]),r=S(r,[1,-1]),[P(s,qt([1,o],s.dtype)),P(qt([a,1],r.dtype),r)])}/**
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
 */function Gf(e,t){let n=f(e,"a","minimum"),s=f(t,"b","minimum");[n,s]=U(n,s),n.dtype==="bool"&&(n=H(n,"int32"),s=H(s,"int32")),G(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Vi,r)}const ya=g({minimum_:Gf});/**
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
 */function Mf(e,t,n){d(n==="reflect"||n==="symmetric",()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);const s=f(e,"x","mirrorPad");if(s.rank===0)throw new Error("mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad");d(t.length===s.rank,()=>`Padding doesn't match input. Must be ${s.rank}. Got ${t.length}.`);const r=n==="reflect"?1:0;for(let i=0;i<s.rank;i++)d(t[i].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),d(t[i][0]>=0&&t[i][0]<=s.shape[i]-r&&t[i][1]>=0&&t[i][1]<=s.shape[i]-r,()=>`Padding in dimension ${i} cannot be greater than or equal to ${s.shape[i]-r} or less than 0 for input of shape ${s.shape}`);const a={paddings:t,mode:n},o={x:s};return b.runKernel(ji,o,a)}const Yf=g({mirrorPad_:Mf});/**
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
 */function Xf(e,t){let n=f(e,"a","mod"),s=f(t,"b","mod");[n,s]=U(n,s);const r={a:n,b:s};return b.runKernel(Ki,r)}const Jf=g({mod_:Xf});/**
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
 */function Zf(e,t=null,n=!1){e=f(e,"x","moments");const s=Ce(t,e.shape),r=Je(e,s,n);let a=r.shape;n||(a=pn(r.shape,s));const o=hn(O(H(e,"float32"),S(r,a))),i=Je(o,s,n);return{mean:r,variance:i}}const Qf=g({moments_:Zf});function tm(e,t,n,s){const r=f(t,"data","multiRNNCell"),a=ve(n,"c","multiRNNCell"),o=ve(s,"h","multiRNNCell");let i=r;const u=[];for(let h=0;h<e.length;h++){const m=e[h](i,a[h],o[h]);u.push(m[0]),u.push(m[1]),i=m[1]}const c=[],p=[];for(let h=0;h<u.length;h+=2)c.push(u[h]),p.push(u[h+1]);return[c,p]}const em=g({multiRNNCell_:tm});/**
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
 */function nm(e,t,n,s=!1){const r=f(e,"logits","multinomial"),a=r.size,o=r.rank;if(a<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${a}.`);if(o>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${o}`);n=n||Math.random();const u={logits:o===1?S(r,[1,-1]):r},c={numSamples:t,seed:n,normalized:s},p=b.runKernel(Ui,u,c);return o===1?S(p,[p.size]):p}const sm=g({multinomial_:nm});/**
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
 */function rm(e,t){let n=f(e,"a","notEqual","string_or_numeric"),s=f(t,"b","notEqual","string_or_numeric");[n,s]=U(n,s),G(n.shape,s.shape);const r={a:n,b:s};return b.runKernel(Hi,r)}const ba=g({notEqual_:rm});/**
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
 */function am(e){const n={x:f(e,"x","onesLike")};return b.runKernel(Xi,n)}const om=g({onesLike_:am});function im(e,t){const n=f(e,"v1","outerProduct"),s=f(t,"v2","outerProduct");d(n.rank===1&&s.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${n.rank} and ${s.rank}.`);const r=S(n,[-1,1]),a=S(s,[1,-1]);return P(r,a)}const um=g({outerProduct_:im});/**
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
 */function cm(e,t,n=0){const s=f(e,"x","pad");if(s.rank===0)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const r={paddings:t,constantValue:n},a={x:s};return b.runKernel(Qi,a,r)}const Le=g({pad_:cm});function lm(e,t,n=0){return d(t.length===2,()=>"Invalid number of paddings. Must be length of 2."),Le(e,[t],n)}const pm=g({pad1d_:lm});function hm(e,t,n=0){return d(t.length===2&&t[0].length===2&&t[1].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),Le(e,t,n)}const fm=g({pad2d_:hm});function mm(e,t,n=0){return d(t.length===3&&t[0].length===2&&t[1].length===2&&t[2].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),Le(e,t,n)}const dm=g({pad3d_:mm});function gm(e,t,n=0){return d(t.length===4&&t[0].length===2&&t[1].length===2&&t[2].length===2&&t[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),Le(e,t,n)}const ym=g({pad4d_:gm});/**
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
 */function bm(e,t,n){const s=f(e,"x","spaceToBatchND");d(s.rank>=1+t.length,()=>`input rank ${s.rank} should be > than [blockShape] ${t.length}`),d(n.length===t.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${t.length}`),d(s.shape.reduce((o,i,u)=>u>0&&u<=t.length?o&&(i+n[u-1][0]+n[u-1][1])%t[u-1]===0:o,!0),()=>`input spatial dimensions ${s.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${t.toString()}`);const r={x:s},a={blockShape:t,paddings:n};return b.runKernel(Iu,r,a)}const wa=g({spaceToBatchND_:bm});/**
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
 */function wm(e,t,n,s,r,a,o){r==null&&(r=[1,1]),a==null&&(a=1),s===0&&(s="valid");const i=f(e,"x","maxPool");let u=i,c=!1;i.rank===3&&(c=!0,u=S(i,[1,i.shape[0],i.shape[1],i.shape[2]])),d(Rt(a,r),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${a} and dilations '${r}'`);const p=fp(u.shape,t,a,r,s),h=[p.dilationHeight,p.dilationWidth];let m;s==="same"?m=Tm([p.filterHeight,p.filterWidth],h):m=[[0,0],[0,0]];const y=h[0]===1&&h[1]===1,[T,N]=Nm([p.inHeight,p.inWidth],h,m),w=y?s:"valid",k=y?u:wa(u,h,T),$=(n==="avg"?()=>Zr(k,t,a,w,o):()=>ga(k,t,a,w,o))(),E=y?$:Qr($,h,N);return c?S(E,[E.shape[1],E.shape[2],E.shape[3]]):E}function Nm(e,t,n){const s=n.map(p=>p[0]),r=n.map(p=>p[1]),a=e.concat(s,r),o=t.map((p,h)=>(p-a[h]%p)%p),i=r.map((p,h)=>p+o[h]),u=t.map((p,h)=>[s[h],i[h]]),c=t.map((p,h)=>[0,o[h]]);return[u,c]}function Tm(e,t){const s=e.map((o,i)=>o+(o-1)*(t[i]-1)).map(o=>o-1),r=s.map(o=>Math.floor(o/2)),a=s.map((o,i)=>o-r[i]);return s.map((o,i)=>[r[i],a[i]])}const Sm=g({pool_:wm});/**
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
 */function km(e,t){const n=f(e,"x","prelu"),s=f(t,"alpha","prelu"),r={x:n,alpha:s};return b.runKernel(eu,r)}const Na=g({prelu_:km});/**
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
 */function $m(e,t=null,n=!1){let s=f(e,"x","prod");s.dtype==="bool"&&(s=H(s,"int32"));const r={x:s},a={axis:t,keepDims:n};return b.runKernel(nu,r,a)}const Em=g({prod_:$m});/**
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
 */function _m(e,t,n,s){const r=e.map((p,h)=>f(p,`tensors${h}`,"raggedGather","int32")),a=f(t,"paramsDenseValues","raggedGather"),o=f(n,"indices","raggedGather","int32"),i={paramsNestedSplits:r,paramsDenseValues:a,indices:o},u={outputRaggedRank:s},c=b.runKernel(su,i,u);return{outputNestedSplits:c.slice(0,c.length-1),outputDenseValues:c[c.length-1]}}const vm=g({raggedGather_:_m});/**
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
 */function Im(e,t,n,s,r){const a=f(e,"shape","raggedTensorToTensor","int32"),o=f(t,"values","raggedTensorToTensor"),i=f(n,"defaultValue","raggedTensorToTensor",o.dtype),u=s.map((h,m)=>f(h,`tensors${m}`,"raggedTensorToTensor","int32")),c={shape:a,values:o,defaultValue:i,rowPartitionTensors:u},p={rowPartitionTypes:r};return b.runKernel(ru,c,p)}const xm=g({raggedTensorToTensor_:Im});/**
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
 */function Dm(e,t,n){const s=q(e);let r=null;if(n==null||n==="float32")r=new Float32Array(s);else if(n==="int32")r=new Int32Array(s);else if(n==="bool")r=new Uint8Array(s);else throw new Error(`Unknown data type ${n}`);for(let a=0;a<s;a++)r[a]=t();return b.makeTensor(r,e,n)}const Am=g({rand_:Dm});/**
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
 */class Ts{constructor(t,n,s,r,a){this.mean=t,this.stdDev=n,this.dtype=s,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);const o=a||Math.random();this.random=ns.alea(o.toString())}nextValue(){if(!isNaN(this.nextVal)){const r=this.nextVal;return this.nextVal=NaN,r}let t,n,s=!1;for(;!s;){let r,a,o;do r=2*this.random()-1,a=2*this.random()-1,o=r*r+a*a;while(o>=1||o===0);const i=Math.sqrt(-2*Math.log(o)/o);t=this.mean+this.stdDev*r*i,n=this.mean+this.stdDev*a*i,(!this.truncated||this.isValidTruncated(t))&&(s=!0)}return(!this.truncated||this.isValidTruncated(n))&&(this.nextVal=this.convertValue(n)),this.convertValue(t)}convertValue(t){return this.dtype==null||this.dtype==="float32"?t:Math.round(t)}isValidTruncated(t){return t<=this.upper&&t>=this.lower}}class Om{constructor(t,n,s,r){this.alpha=t,this.beta=1/n,this.dtype=s;const a=r||Math.random();this.randu=ns.alea(a.toString()),this.randn=new Ts(0,1,s,!1,this.randu()),t<1?this.d=t+2/3:this.d=t-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let t,n,s,r,a,o;for(;;){do r=this.randn.nextValue(),o=1+this.c*r;while(o<=0);if(o*=o*o,t=r*r,n=1-.331*t*t,s=.5*t+this.d*(1-o+Math.log(o)),a=this.randu(),a<n||Math.log(a)<s)break}return o=1/this.beta*this.d*o,this.alpha<1&&(o*=Math.pow(this.randu(),1/this.alpha)),this.convertValue(o)}convertValue(t){return this.dtype==="float32"?t:Math.round(t)}}class Cm{constructor(t=0,n=1,s,r){if(this.canReturnFloat=()=>this.dtype==null||this.dtype==="float32",this.min=t,this.range=n-t,this.dtype=s,r==null&&(r=Math.random()),typeof r=="number"&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${t} - ${n} <= 1 and dtype is not float`);this.random=ns.alea(r)}convertValue(t){return this.canReturnFloat()?t:Math.round(t)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}/**
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
 */function Fm(e,t,n=1,s="float32",r){if(n==null&&(n=1),s==null&&(s="float32"),s!=="float32"&&s!=="int32")throw new Error(`Unsupported data type ${s}`);const a=new Om(t,n,s,r),o=_t(e,s);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const Bm=g({randomGamma_:Fm});/**
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
 */function Pm(e,t=0,n=1,s,r){if(s!=null&&s==="bool")throw new Error(`Unsupported data type ${s}`);const a=new Ts(t,n,s,!1,r),o=_t(e,s);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const Ta=g({randomNormal_:Pm});/**
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
 */function Lm(e,t,n){if(t!=null&&t==="bool")throw new Error(`Unsupported data type ${t}`);return Ta(e,0,1,t,n)}const Rm=g({randomStandardNormal_:Lm});/**
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
 */function zm(e,t=0,n=1,s="float32",r){const a=_t(e,s),o=new Cm(t,n,null,r);for(let i=0;i<a.values.length;i++)a.values[i]=o.nextValue();return a.toTensor()}const Sa=g({randomUniform_:zm});/**
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
 */function De(e,t,n=1,s="float32"){if(n===0)throw new Error("Cannot have a step of zero");const r={start:e,stop:t,step:n,dtype:s};return b.runKernel(au,{},r)}/**
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
 */function Vm(e){const n={x:f(e,"x","reciprocal")};return b.runKernel(iu,n)}const jm=g({reciprocal_:Vm});/**
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
 */function Km(e){const n={x:f(e,"x","relu")};return b.runKernel(uu,n)}const gn=g({relu_:Km});/**
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
 */function Um(e){const n={x:f(e,"x","relu6")};return b.runKernel(hu,n)}const ka=g({relu6_:Um});/**
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
 */function Wm(e,t){const s={x:f(e,"x","reverse")},r={dims:t};return b.runKernel(fu,s,r)}const Qt=g({reverse_:Wm});/**
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
 */function qm(e){const t=f(e,"x","reverse");return d(t.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${t.rank}.`),Qt(t,0)}const Hm=g({reverse1d_:qm});/**
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
 */function Gm(e,t){const n=f(e,"x","reverse");return d(n.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${n.rank}.`),Qt(n,t)}const Mm=g({reverse2d_:Gm});/**
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
 */function Ym(e,t){const n=f(e,"x","reverse");return d(n.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${n.rank}.`),Qt(n,t)}const Xm=g({reverse3d_:Ym});/**
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
 */function Jm(e,t){const n=f(e,"x","reverse");return d(n.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${n.rank}.`),Qt(n,t)}const Zm=g({reverse4d_:Jm});/**
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
 */function Qm(e){const n={x:f(e,"x","round")};return b.runKernel(mu,n)}const $a=g({round_:Qm});/**
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
 */function td(e){const n={x:f(e,"x","rsqrt","float32")};return b.runKernel(du,n)}const ed=g({rsqrt_:td});/**
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
 */function nd(e){const n={x:f(e,"x","selu")};return b.runKernel(wu,n)}const sd=g({selu_:nd});function rd(e,t,n,s,r,a=[1,1],o="NHWC"){const i=f(e,"x","separableConv2d"),u=f(t,"depthwiseFilter","separableConv2d"),c=f(n,"pointwiseFilter","separableConv2d");let p=i,h=!1;if(i.rank===3&&(h=!0,p=S(i,[1,i.shape[0],i.shape[1],i.shape[2]])),o==="NCHW")throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");d(p.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${p.rank}.`),d(u.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${u.rank}.`),d(c.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${u.rank}.`),d(c.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${c.shape[0]}.`),d(c.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${c.shape[1]}.`);const m=u.shape[2],y=u.shape[3];d(c.shape[2]===m*y,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${m*y}, but got ${c.shape[2]}.`);const T=gs(p,u,s,r,o,a),w=ln(T,c,1,"valid",o);return h?S(w,[w.shape[1],w.shape[2],w.shape[3]]):w}const ad=g({separableConv2d_:rd});/**
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
 */async function od(e,t){const n=f(e,"x","setdiff1d"),s=f(t,"y","setdiff1d");d(n.dtype===s.dtype,()=>`x and y should have the same dtype, but got x (${n.dtype}) and y (${s.dtype}).`),d(n.rank===1,()=>`x should be 1D tensor, but got x (${n.shape}).`),d(s.rank===1,()=>`y should be 1D tensor, but got y (${s.shape}).`);const r=await n.data(),a=await s.data(),o=new Set(a);let i=0;for(let p=0;p<r.length;p++)o.has(r[p])||i++;const u=new xn([i],n.dtype),c=new xn([i],"int32");for(let p=0,h=0;p<r.length;p++)o.has(r[p])||(u.values[h]=r[p],c.values[h]=p,h++);return[u.toTensor(),c.toTensor()]}const id=od;/**
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
 */function ud(e){const n={x:f(e,"x","sign")};return b.runKernel(ku,n)}const cd=g({sign_:ud});/**
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
 */function ld(e){const n={x:f(e,"x","sin","float32")};return b.runKernel(Tu,n)}const pd=g({sin_:ld});/**
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
 */function hd(e){const n={x:f(e,"x","sinh")};return b.runKernel(Su,n)}const fd=g({sinh_:hd});/**
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
 */function md(e,t,n){const s=f(e,"x","slice1d");return d(s.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${s.rank} tensor`),R(s,[t],[n])}const dd=g({slice1d_:md});/**
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
 */function gd(e,t,n){const s=f(e,"x","slice2d");return d(s.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${s.rank} tensor`),R(s,t,n)}const yd=g({slice2d_:gd});/**
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
 */function bd(e,t,n){const s=f(e,"x","slice3d");return d(s.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${s.rank} tensor`),R(s,t,n)}const wd=g({slice3d_:bd});/**
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
 */function Nd(e,t,n){const s=f(e,"x","slice4d");return d(s.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${s.rank} tensor`),R(s,t,n)}const Td=g({slice4d_:Nd});/**
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
 */function Sd(e,t=-1){const n=f(e,"logits","softmax","float32");if(t===-1&&(t=n.rank-1),t!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${t}`);const s={logits:n},r={dim:t};return b.runKernel(Du,s,r)}const kd=g({softmax_:Sd});/**
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
 */function $d(e){d(e.dtype==="complex64",()=>`The dtype for tf.spectral.fft() must be complex64 but got ${e.dtype}.`);const t={input:e};return b.runKernel(ui,t)}const Ss=g({fft_:$d});/**
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
 */function Ed(e){d(e.dtype==="complex64",()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${e.dtype}.`);const t={input:e};return b.runKernel(bi,t)}const Ze=g({ifft_:Ed});/**
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
 */function _d(e){const t=e.shape[e.shape.length-1],n=e.size/t;let s;if(t<=2){const r=S(e,[n,t]);s=Ze(r)}else{const r=[n,2*(t-1)],a=S(Ie(e),[n,t]),o=S(an(e),[n,t]),i=Qt(R(a,[0,1],[n,t-2]),1),u=D(Qt(R(o,[0,1],[n,t-2]),1),B(-1)),c=rt([a,i],1),p=rt([o,u],1),h=S(Bt(c,p),[r[0],r[1]]);s=Ze(h)}if(s=Ie(s),e.rank===3&&e.shape[0]!==0){const r=s,a=e.shape[0];s=S(s,[a,s.shape[0]/a,s.shape[1]]),r.dispose()}return s}const Ea=g({irfft_:_d});/**
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
 */function vd(e,t,n=0){const r={x:f(e,"x","split")},a={numOrSizeSplits:t,axis:n};return b.runKernel(xu,r,a)}const Ae=g({split_:vd});/**
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
 */function Id(e,t){d(e.dtype==="float32",()=>`The dtype for rfft() must be real value but got ${e.dtype}`);let n=e.shape[e.shape.length-1];const s=e.size/n;let r;if(t!=null&&t<n){const T=e.shape.map(w=>0),N=e.shape.map(w=>w);N[e.shape.length-1]=t,r=R(e,T,N),n=t}else if(t!=null&&t>n){const T=e.shape.map(N=>N);T[e.shape.length-1]=t-n,r=rt([e,ge(T)],e.shape.length-1),n=t}else r=e;const a=ys(r),o=S(Bt(r,a),[s,n]),i=Ss(o),u=Math.floor(n/2)+1,c=Ie(i),p=an(i),h=Ae(c,[u,n-u],c.shape.length-1),m=Ae(p,[u,n-u],p.shape.length-1),y=r.shape.slice();return y[r.shape.length-1]=u,S(Bt(h[0],m[0]),y)}const ks=g({rfft_:Id});/**
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
 */function xd(e,t){let n=f(e,"a","squaredDifference"),s=f(t,"b","squaredDifference");[n,s]=U(n,s),G(n.shape,s.shape);const r={a:n,b:s},a={};return b.runKernel(Pu,r,a)}const _a=g({squaredDifference_:xd});/**
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
 */function Dd(e,t){const n=f(e,"x","squeeze","string_or_numeric");return S(n,ar(n.shape,t).newShape)}const $s=g({squeeze_:Dd});/**
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
 */function Ad(e,t=0){const n=ve(e,"tensors","stack","string_or_numeric");d(n.length>=1,()=>"Pass at least one tensor to tf.stack"),n.length>0&&d(t<=n[0].rank,()=>"Axis must be <= rank of the tensor");const s=n,r={axis:t};return b.runKernel(Zi,s,r)}const It=g({stack_:Ad});/**
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
 */function Od(e,t=0){const s={x:f(e,"x","step")},r={alpha:t};return b.runKernel(Xu,s,r)}const va=g({step_:Od});/**
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
 */function Cd(e,t,n,s,r=0,a=0,o=0,i=0,u=0){const p={x:f(e,"x","stridedSlice","string_or_numeric")},h={begin:t,end:n,strides:s,beginMask:r,endMask:a,ellipsisMask:o,newAxisMask:i,shrinkAxisMask:u};return b.runKernel(Lu,p,h)}const Fd=g({stridedSlice_:Cd});/**
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
 */function Bd(e){const n={x:f(e,"x","tan","float32")};return b.runKernel(Ku,n)}const Pd=g({tan_:Bd});/**
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
 */function dt(e,t){ee(e);const n=Pt(e,t);if(n.length!==1)throw new Error("tensor1d() requires values to be a flat/TypedArray");return Lt(e,null,n,t)}/**
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
 */function $e(e,t,n){if(ee(e),t!=null&&t.length!==2)throw new Error("tensor2d() requires shape to have two numbers");const s=Pt(e,n);if(s.length!==2&&s.length!==1)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return Lt(e,t,s,n)}/**
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
 */function Ld(e,t,n){if(ee(e),t!=null&&t.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const s=Pt(e,n);if(s.length!==4&&s.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return Lt(e,t,s,n)}/**
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
 */function Rd(e,t,n){if(ee(e),t!=null&&t.length!==5)throw new Error("tensor5d() requires shape to have five numbers");const s=Pt(e,n);if(s.length!==5&&s.length!==1)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return Lt(e,t,s,n)}/**
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
 */function zd(e,t,n){if(ee(e),t!=null&&t.length!==6)throw new Error("tensor6d() requires shape to have six numbers");const s=Pt(e,n);if(s.length!==6&&s.length!==1)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return t=t||s,Lt(e,t,s,n)}/**
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
 */function Vd(e,t=1,n=!0){const s=f(e,"x","topk");if(s.rank===0)throw new Error("topk() expects the input to be of rank 1 or higher");const r=s.shape[s.shape.length-1];if(t<0)throw new Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>r)throw new Error(`'k' passed to topk() must be <= the last dimension (${r}) but got ${t}`);const a={x:s},o={k:t,sorted:n},[i,u]=b.runKernel(Wu,a,o);return{values:i,indices:u}}const jd=g({topk_:Vd});/**
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
 */function Kd(e,t=0,n=1,s,r){if(s!=null&&s==="bool")throw new Error("Unsupported data type $ { dtype }");const a=new Ts(t,n,s,!0,r),o=_t(e,s);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const Ud=g({truncatedNormal_:Kd});/**
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
 */function Wd(e,t=0){const n=f(e,"x","unique","string_or_numeric");d(n.rank>0,()=>"The input tensor must be at least 1D");const s={x:n},r={axis:t},[a,o]=b.runKernel(Hu,s,r);return{values:a,indices:o}}const qd=g({unique_:Wd});/**
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
 */function Hd(e,t,n){const s=f(e,"x","unsortedSegmentSum"),r=f(t,"segmentIds","unsortedSegmentSum","int32");d(he(n),()=>"numSegments must be of dtype int");const a={x:s,segmentIds:r},o={numSegments:n};return b.runKernel(Mu,a,o)}const Gd=g({unsortedSegmentSum_:Hd});/**
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
 */function Md(e,t=0){const n=f(e,"x","unstack","string_or_numeric");d(t>=-n.shape.length&&t<n.shape.length,()=>`Axis = ${t} is not in [-${n.shape.length}, ${n.shape.length})`);const s={value:n},r={axis:t};return b.runKernel(Gu,s,r)}const ne=g({unstack_:Md});/**
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
 */function Yd(e,t){return Ns(e,t,"right")}/**
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
 */function Xd(e,t=!0,n,s){return b.makeVariable(e,t,n,s)}/**
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
 */function Jd(e,t){const n=[];for(let a=0;a<t.length;a++)t[a]&&n.push(a);const s=_t(e,"int32"),r=_t([n.length,e.length],"int32");for(let a=0;a<n.length;a++){const o=s.indexToLoc(n[a]),i=a*e.length;r.values.set(o,i)}return r.toTensor()}/**
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
 */async function Zd(e){const t=f(e,"condition","whereAsync","bool"),n=await t.data(),s=Jd(t.shape,n);return e!==t&&t.dispose(),s}const Ia=Zd;/**
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
 */async function Qd(e,t,n){const s=f(e,"tensor","boolMask"),r=f(t,"mask","boolMask","bool"),a=n??0,o=r.rank,i=s.shape;d(o>0,()=>"mask cannot be scalar"),at(i.slice(a,a+o),r.shape,"mask's shape must match the first K dimensions of tensor's shape,");let u=1;for(let N=a;N<a+o;N++)u*=i[N];const c=i.slice(0,a).concat([u],i.slice(a+o)),p=S(s,c),h=S(r,[-1]),m=await Ia(h),y=$s(m,[1]),T=ua(p,y,a);return e!==s&&s.dispose(),t!==r&&r.dispose(),y.dispose(),p.dispose(),h.dispose(),m.dispose(),T}const tg=Qd;/**
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
 */function eg(e,t,n,s,r=!0){const a=f(e,"v","movingAverage"),o=f(t,"x","movingAverage"),i=f(n,"decay","movingAverage");$r(a,o),d(Et(a.shape,o.shape),()=>"Shape mismatch in v and x");const u=B(1),c=O(u,i);let p=D(O(o,a),c);if(r){d(s!=null,()=>"When using zeroDebias: true, step is required.");const h=f(s,"step","movingAverage");p=st(p,O(u,bs(i,h)))}return Y(a,p)}const ng=g({movingAverage_:eg});/**
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
 */function sg(e,t,n){const s=f(e,"indices","scatterND","int32"),r=f(t,"updates","scatterND");Yr(r,s,n);const a={indices:s,updates:r},o={shape:n};return b.runKernel(gu,a,o)}const rg=g({scatterND_:sg});function ag(e,t,n,s){if(e.dtype!=="int32")throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${e.shape}.`);const r=e.rank>0?e.shape[0]:1,a=e.rank>1?e.shape[1]:1;if(n.length!==a)throw new Error(`outputShape has incorrect number of elements:, ${n.length}, should be: ${a}.`);const o=t.size;if(!(t.rank===0||t.rank===1&&o===r))throw new Error(`sparseValues has incorrect shape ${t.shape}, should be [] or [${r}]`);if(t.dtype!==s.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}/**
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
 */function og(e,t,n,s=0){const r=f(e,"sparseIndices","sparseToDense","int32"),a=f(t,"sparseValues","sparseToDense","string_or_numeric"),o=f(s,"defaultValue","sparseToDense",a.dtype);ag(r,a,n,o);const i={sparseIndices:r,sparseValues:a,defaultValue:o},u={outputShape:n};return b.runKernel(Bu,i,u)}const ig=g({sparseToDense_:og});/**
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
 */function ug(e,t){const n=f(t,"indices","gatherND","int32"),r={params:f(e,"x","gatherND","string_or_numeric"),indices:n};return b.runKernel(di,r)}const cg=g({gatherND_:ug});/**
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
 */function lg(e,t){if(t==null)return e.shape.slice();if(Et(e.shape,t))return t;if(e.shape.length===t.length){const n=[];for(let s=0;s<e.shape.length;s++)t[s]==null&&e.shape[s]!=null?n.push(e.shape[s]):n.push(t[s]);return n}return t}/**
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
 */function pg(e,t,n,s){const r=f(e,"x","dropout");if(d(r.dtype==="float32",()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${r.dtype} tensor instead.`),d(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),t===0)return e instanceof W?r.clone():r;const a=lg(r,n),o=1-t,i=st(ia(Y(Sa(a,0,1,"float32",s),o)),o);return D(r,i)}const hg=g({dropout_:pg});/**
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
 */function xa(e){return Math.floor(Math.pow(2,Math.ceil(Math.log(e)/Math.log(2))))}function Es(e,t,n){const s=1-e%2,r=new Float32Array(e);for(let a=0;a<e;++a){const o=2*Math.PI*a/(e+s-1);r[a]=t-n*Math.cos(o)}return dt(r,"float32")}/**
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
 */async function fg(e,t,n=1){const s=f(e,"predictions","inTopK"),r=f(t,"targets","inTopK");d(s.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${s.rank}`),d(s.rank-1===r.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${s.rank} and targets rank ${r.rank}`),at(s.shape.slice(0,s.shape.length-1),r.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const a=s.shape[s.shape.length-1];d(n>0&&n<=a,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${a}), but got ${n}`);const o=await s.data(),i=await r.data(),[u,c]=[o.length/a,a],p=or("bool",u);for(let h=0;h<u;h++){const m=h*c,y=o.subarray(m,m+c),T=[];for(let N=0;N<y.length;N++)T.push({value:y[N],index:N});T.sort((N,w)=>w.value-N.value),p[h]=0;for(let N=0;N<n;N++)if(T[N].index===i[h]){p[h]=1;break}}return e!==s&&s.dispose(),t!==r&&r.dispose(),Nt(p,r.shape,"bool")}const mg=fg;/**
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
 */function dg(e,t,n,s,r,a="NHWC",o){let i=e;e.rank===3&&(i=S(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let u=t;u.rank===3&&(u=S(t,[1,t.shape[0],t.shape[1],t.shape[2]])),d(i.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${i.shape}.`),d(u.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${u.shape}.`),d(n.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);const c=a==="NHWC"?i.shape[3]:i.shape[1],p=a==="NHWC"?u.shape[3]:u.shape[1];d(c===n[2],()=>`Error in conv2dDerFilter: depth of input ${c}) must match input depth in filter (${n[2]}.`),d(p===n[3],()=>`Error in conv2dDerFilter: depth of dy (${p}) must match output depth for filter (${n[3]}).`),bt("conv2dDerFilter",r,o);const h={x:i,dy:u},m={strides:s,pad:r,dataFormat:a,dimRoundingMode:o,filterShape:n};return b.runKernel(Ro,h,m)}const gg=g({conv2DBackpropFilter_:dg});/**
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
 */function _s(e,t,n){if(n==null||n==="linear")return e;if(n==="relu")return D(e,va(t));throw new Error(`Cannot compute gradient for fused activation ${n}.`)}function vs(e,t){let n=t;const s=qr(e.shape,t.shape);return s.length>0&&(n=j(n,s)),S(n,e.shape)}function Is(e,t,n,s){if(t==="linear")return e;if(t==="relu")return gn(e);if(t==="elu")return sa(e);if(t==="relu6")return ka(e);if(t==="prelu")return Na(e,n);if(t==="leakyrelu")return la(e,s);if(t==="sigmoid")return le(e);throw new Error(`Unknown fused activation ${t}.`)}const xs=(e,t)=>!(e>0)||t==="linear";/**
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
 */function yg({x:e,filter:t,strides:n,pad:s,dataFormat:r="NHWC",dilations:a=[1,1],dimRoundingMode:o,bias:i,activation:u="linear",preluActivationWeights:c,leakyreluAlpha:p}){if(u=u||"linear",xs(b.state.gradientDepth,u)===!1){d(r==="NHWC",()=>`Error in fused conv2d: got dataFormat of ${r} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let _=ln(e,t,n,s,r,a,o);return i!=null&&(_=Y(_,i)),Is(_,u,c,p)}const h=f(e,"x","conv2d","float32"),m=f(t,"filter","conv2d","float32");let y=h,T=!1;h.rank===3&&(T=!0,y=S(h,[1,h.shape[0],h.shape[1],h.shape[2]])),d(y.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${y.rank}.`),d(m.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${m.rank}.`),bt("fused conv2d",s,o);const N=r==="NHWC"?y.shape[3]:y.shape[1];d(m.shape[2]===N,()=>`Error in conv2d: depth of input (${N}) must match input depth for filter ${m.shape[2]}.`),d(Rt(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);const w=on(y.shape,m.shape,n,a,s,o);let k;i!=null&&(k=f(i,"bias","fused conv2d"),[k]=U(k,h),r==="NHWC"?G(w.outShape,k.shape):(d(k.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${k.shape.length}.`),d(k.shape.length===0||k.shape[0]===w.outChannels||k.shape[0]===1,()=>`Error in fused conv2d: bias shape (${k.shape}) is not compatible with the number of output channels (${w.outChannels})`)));let v;if(c!=null){const _=c.shape;if(d(_.length<=1||_.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${_.length}.`),_.length===1)d(_[0]===1||_[0]===w.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${_}) is not compatible with the number of output channels (${w.outChannels}).`);else if(_.length===3)try{G(_,w.outShape)}catch{const F=`Error in fused conv2d: PReLU activation weights (${_}) is not compatible with the output shape of the conv2d (${w.outShape}).`;throw Error(F)}v=f(c,"prelu weights","fused conv2d")}const $=(_,A)=>{d(r==="NHWC",()=>`Error in gradient of fused conv2D: got dataFormat of ${r} but only NHWC is currently supported.`);const[F,C,z,V]=A,Z=_s(_,z,u);d(Ye(a),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${a}'`);const X=ea(C.shape,Z,F,n,s),Q=gg(C,Z,F.shape,n,s),it=[X,Q];if(V!=null){const se=vs(V,Z);it.push(se)}return it},E={x:y,filter:m,bias:k,preluActivationWeights:v},I={strides:n,pad:s,dataFormat:r,dilations:a,dimRoundingMode:o,activation:u,leakyreluAlpha:p};return i==null?vt((A,F,C)=>{let z=b.runKernel(Ps,E,I);return C([F,A,z]),T&&(z=S(z,[z.shape[1],z.shape[2],z.shape[3]])),{value:z,gradFunc:$}})(y,m):vt((A,F,C,z)=>{let V=b.runKernel(Ps,E,I);return z([F,A,V,C]),T&&(V=S(V,[V.shape[1],V.shape[2],V.shape[3]])),{value:V,gradFunc:$}})(y,m,k)}const bg=g({fusedConv2d_:yg});/**
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
 */function wg(e,t,n,s,r,a=[1,1],o){let i=e;e.rank===3&&(i=S(e,[1,e.shape[0],e.shape[1],e.shape[2]]));let u=t;u.rank===3&&(u=S(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const c={x:i,dy:u},p={strides:s,pad:r,dimRoundingMode:o,dilations:a,filterShape:n};return b.runKernel(Xo,c,p)}const Ng=g({depthwiseConv2dNativeBackpropFilter_:wg});/**
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
 */function Tg(e,t,n,s,r,a=[1,1],o){let i=t,u=!1;t.rank===3&&(u=!0,i=S(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const c={dy:i,filter:n},p={strides:s,pad:r,dimRoundingMode:o,dilations:a,inputShape:e},h=b.runKernel(Jo,c,p);return u?S(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const Sg=g({depthwiseConv2dNativeBackpropInput_:Tg});/**
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
 */function kg({x:e,filter:t,strides:n,pad:s,dataFormat:r="NHWC",dilations:a=[1,1],dimRoundingMode:o,bias:i,activation:u="linear",preluActivationWeights:c,leakyreluAlpha:p}){if(xs(b.state.gradientDepth,u)===!1){let I=gs(e,t,n,s,r,a,o);return i!=null&&(I=Y(I,i)),Is(I,u,c,p)}const h=f(e,"x","depthwiseConv2d","float32"),m=f(t,"filter","depthwiseConv2d","float32");let y=h,T=!1;h.rank===3&&(T=!0,y=S(h,[1,h.shape[0],h.shape[1],h.shape[2]])),d(y.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${y.rank}.`),d(m.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${m.rank}.`),d(y.shape[3]===m.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${y.shape[3]}) must match the inChannels dimension in filter ${m.shape[2]}.`),a==null&&(a=[1,1]),d(Rt(n,a),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),bt("fused depthwiseConv2d",s,o);const N=on(y.shape,m.shape,n,a,s,o,!0);let w;i!=null&&(w=f(i,"bias","fused conv2d"),[w]=U(w,h),G(N.outShape,w.shape));let k;c!=null&&(k=f(c,"prelu weights","fused depthwiseConv2d"));const v=(I,_)=>{d(Ye(a),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${a}'`);const[A,F,C,z]=_,V=_s(I,C,u),Z=Sg(F.shape,V,A,n,s,a,o),X=Ng(F,V,A.shape,n,s,a,o);if(z!=null){const Q=vs(w,V);return[Z,X,Q]}return[Z,X]},$={x:y,filter:m,bias:w,preluActivationWeights:k},E={strides:n,pad:s,dataFormat:r,dilations:a,dimRoundingMode:o,activation:u,leakyreluAlpha:p};return i==null?vt((_,A,F)=>{let C=b.runKernel(Ls,$,E);return F([A,_,C]),T&&(C=S(C,[C.shape[1],C.shape[2],C.shape[3]])),{value:C,gradFunc:v}})(y,m):vt((_,A,F,C)=>{let z=b.runKernel(Ls,$,E);return C([A,_,z,F]),T&&(z=S(z,[z.shape[1],z.shape[2],z.shape[3]])),{value:z,gradFunc:v}})(y,m,w)}const $g=g({fusedDepthwiseConv2d_:kg});/**
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
 */function Eg({a:e,b:t,transposeA:n=!1,transposeB:s=!1,bias:r,activation:a="linear",preluActivationWeights:o,leakyreluAlpha:i=.2}){if(xs(b.state.gradientDepth,a)===!1){let V=P(e,t,n,s);return r!=null&&(V=Y(V,r)),Is(V,a,o,i)}let u=f(e,"a","fused matMul"),c=f(t,"b","fused matMul");[u,c]=U(u,c);const p=n?u.shape[u.rank-2]:u.shape[u.rank-1],h=s?c.shape[c.rank-1]:c.shape[c.rank-2],m=n?u.shape[u.rank-1]:u.shape[u.rank-2],y=s?c.shape[c.rank-2]:c.shape[c.rank-1],T=u.shape.slice(0,-2),N=c.shape.slice(0,-2),w=q(T),k=q(N);d(p===h,()=>`Error in fused matMul: inner shapes (${p}) and (${h}) of Tensors with shapes ${u.shape} and ${c.shape} and transposeA=${n} and transposeB=${s} must match.`);const $=G(u.shape.slice(0,-2),c.shape.slice(0,-2)).concat([m,y]),E=n?S(u,[w,p,m]):S(u,[w,m,p]),I=s?S(c,[k,y,h]):S(c,[k,h,y]);let _;r!=null&&(_=f(r,"bias","fused matMul"),[_]=U(_,u),G($,_.shape));let A;o!=null&&(A=f(o,"prelu weights","fused matMul"));const F=(V,Z)=>{const[X,Q,it,se]=Z,Tt=_s(S(V,it.shape),it,a);let re,ae;if(!n&&!s?(re=P(Tt,Q,!1,!0),ae=P(X,Tt,!0,!1)):!n&&s?(re=P(Tt,Q,!1,!1),ae=P(Tt,X,!0,!1)):n&&!s?(re=P(Q,Tt,!1,!0),ae=P(X,Tt,!1,!1)):(re=P(Q,Tt,!0,!0),ae=P(Tt,X,!0,!0)),r!=null){const La=vs(se,Tt);return[re,ae,La]}else return[re,ae]},C={a:E,b:I,bias:_,preluActivationWeights:A},z={transposeA:n,transposeB:s,activation:a,leakyreluAlpha:i};return r==null?vt((Z,X,Q)=>{const it=b.runKernel(Bs,C,z);return Q([Z,X,it]),{value:S(it,$),gradFunc:F}})(E,I):vt((Z,X,Q,it)=>{const se=b.runKernel(Bs,C,z);return it([Z,X,se,Q]),{value:S(se,$),gradFunc:F}})(E,I,_)}const _g=g({fusedMatMul_:Eg});/**
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
 */const vg=Object.freeze(Object.defineProperty({__proto__:null,conv2d:bg,depthwiseConv2d:$g,matMul:_g},Symbol.toStringTag,{value:"Module"}));/**
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
 */function Ig(e){return Es(e,.54,.46)}const xg=g({hammingWindow_:Ig});/**
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
 */function Dg(e){return Es(e,.5,.5)}const Da=g({hannWindow_:Dg});/**
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
 */function Ag(e,t,n,s=!1,r=0){let a=0;const o=[];for(;a+t<=e.size;)o.push(R(e,a,t)),a+=n;if(s)for(;a<e.size;){const i=a+t-e.size,u=rt([R(e,a,t-i),cn([i],r)]);o.push(u),a+=n}return o.length===0?$e([],[0,t]):S(rt(o),[o.length,t])}const Aa=g({frame_:Ag});/**
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
 */function Og(e,t,n,s,r=Da){s==null&&(s=xa(t));const a=Aa(e,t,n),o=D(a,r(t));return ks(o,s)}const Cg=g({stft_:Og});/**
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
 */function Fg(e,t,n,s,r="bilinear",a=0){const o=f(e,"image","cropAndResize"),i=f(t,"boxes","cropAndResize","float32"),u=f(n,"boxInd","cropAndResize","int32"),c=i.shape[0];d(o.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${o.rank}.`),d(i.rank===2&&i.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${c},4] but had shape ${i.shape}.`),d(u.rank===1&&u.shape[0]===c,()=>`Error in cropAndResize: boxInd must be have size [${c}] but had shape ${i.shape}.`),d(s.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${s.length}.`),d(s[0]>=1&&s[1]>=1,()=>`cropSize must be atleast [1,1], but was ${s}`),d(r==="bilinear"||r==="nearest",()=>`method must be bilinear or nearest, but was ${r}`);const p={image:o,boxes:i,boxInd:u},h={method:r,extrapolationValue:a,cropSize:s};return b.runKernel(Ho,p,h)}const Bg=g({cropAndResize_:Fg});/**
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
 */function Pg(e){const t=f(e,"image","flipLeftRight","float32");d(t.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);const n={image:t};return b.runKernel(li,n,{})}const Lg=g({flipLeftRight_:Pg});/**
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
 */function Rg(e){const t=f(e,"image","grayscaleToRGB"),n=t.rank-1,s=t.shape[n];d(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),d(s===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${s}.`);const r=new Array(t.rank);return r.fill(1,0,n),r[n]=3,ke(t,r)}const zg=g({grayscaleToRGB_:Rg});/**
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
 */function Vg(e,t,n=0,s=.5){const r=f(e,"image","rotateWithOffset","float32");d(r.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${r.rank}.`);const a={image:r},o={radians:t,fillValue:n,center:s};return b.runKernel(Ju,a,o)}const jg=g({rotateWithOffset_:Vg});/**
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
 */function ye(e,t,n,s,r,a){s==null&&(s=.5),r==null&&(r=Number.NEGATIVE_INFINITY),a==null&&(a=0);const o=e.shape[0];return n=Math.min(n,o),d(0<=s&&s<=1,()=>`iouThreshold must be in [0, 1], but was '${s}'`),d(e.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${e.rank}'`),d(e.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${e.shape[1]}`),d(t.rank===1,()=>"scores must be a 1D tensor"),d(t.shape[0]===o,()=>`scores has incompatible shape with boxes. Expected ${o}, but was ${t.shape[0]}`),d(0<=a&&a<=1,()=>`softNmsSigma must be in [0, 1], but was '${a}'`),{maxOutputSize:n,iouThreshold:s,scoreThreshold:r,softNmsSigma:a}}/**
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
 */function Kg(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY){const a=f(e,"boxes","nonMaxSuppression","float32"),o=f(t,"scores","nonMaxSuppression","float32"),i=ye(a,o,n,s,r);n=i.maxOutputSize,s=i.iouThreshold,r=i.scoreThreshold;const u={maxOutputSize:n,iouThreshold:s,scoreThreshold:r};return b.runKernel(Gi,{boxes:a,scores:o},u)}const Ug=g({nonMaxSuppression_:Kg});/**
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
 */function Wg(e,t,n){const s=qg(e,t,n),r=s<0?-(s+1):s;e.splice(r,0,t)}function qg(e,t,n){return Gg(e,t,n||Hg)}function Hg(e,t){return e>t?1:e<t?-1:0}function Gg(e,t,n){let s=0,r=e.length,a=0,o=!1;for(;s<r;){a=s+(r-s>>>1);const i=n(t,e[a]);i>0?s=a+1:(r=a,o=!i)}return o?s:-s-1}/**
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
 */function Mg(e,t,n,s,r){return Ds(e,t,n,s,r,0)}function Yg(e,t,n,s,r,a){return Ds(e,t,n,s,r,0,!1,a,!0)}function Xg(e,t,n,s,r,a){return Ds(e,t,n,s,r,a,!0)}function Ds(e,t,n,s,r,a,o=!1,i=!1,u=!1){const c=[];for(let w=0;w<t.length;w++)t[w]>r&&c.push({score:t[w],boxIndex:w,suppressBeginIndex:0});c.sort(Ms);const p=a>0?-.5/a:0,h=[],m=[];for(;h.length<n&&c.length>0;){const w=c.pop(),{score:k,boxIndex:v,suppressBeginIndex:$}=w;if(k<r)break;let E=!1;for(let I=h.length-1;I>=$;--I){const _=Jg(e,v,h[I]);if(_>=s){E=!0;break}if(w.score=w.score*Zg(s,p,_),w.score<=r)break}w.suppressBeginIndex=h.length,E||(w.score===k?(h.push(v),m.push(w.score)):w.score>r&&Wg(c,w,Ms))}const y=h.length,T=n-y;i&&T>0&&(h.push(...new Array(T).fill(0)),m.push(...new Array(T).fill(0)));const N={selectedIndices:h};return o&&(N.selectedScores=m),u&&(N.validOutputs=y),N}function Jg(e,t,n){const s=e.subarray(t*4,t*4+4),r=e.subarray(n*4,n*4+4),a=Math.min(s[0],s[2]),o=Math.min(s[1],s[3]),i=Math.max(s[0],s[2]),u=Math.max(s[1],s[3]),c=Math.min(r[0],r[2]),p=Math.min(r[1],r[3]),h=Math.max(r[0],r[2]),m=Math.max(r[1],r[3]),y=(i-a)*(u-o),T=(h-c)*(m-p);if(y<=0||T<=0)return 0;const N=Math.max(a,c),w=Math.max(o,p),k=Math.min(i,h),v=Math.min(u,m),$=Math.max(k-N,0)*Math.max(v-w,0);return $/(y+T-$)}function Zg(e,t,n){const s=Math.exp(t*n*n);return n<=e?s:0}function Ms(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}/**
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
 */async function Qg(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY){const a=f(e,"boxes","nonMaxSuppressionAsync"),o=f(t,"scores","nonMaxSuppressionAsync"),i=ye(a,o,n,s,r);n=i.maxOutputSize,s=i.iouThreshold,r=i.scoreThreshold;const u=await Promise.all([a.data(),o.data()]),c=u[0],p=u[1],{selectedIndices:h}=Mg(c,p,n,s,r);return a!==e&&a.dispose(),o!==t&&o.dispose(),dt(h,"int32")}const ty=Qg;/**
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
 */function ey(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,a=0){const o=f(e,"boxes","nonMaxSuppression"),i=f(t,"scores","nonMaxSuppression"),u=ye(o,i,n,s,r,a);n=u.maxOutputSize,s=u.iouThreshold,r=u.scoreThreshold,a=u.softNmsSigma;const c={boxes:o,scores:i},p={maxOutputSize:n,iouThreshold:s,scoreThreshold:r,softNmsSigma:a},h=b.runKernel(Yi,c,p);return{selectedIndices:h[0],selectedScores:h[1]}}const ny=g({nonMaxSuppressionWithScore_:ey});/**
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
 */async function sy(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,a=0){const o=f(e,"boxes","nonMaxSuppressionAsync"),i=f(t,"scores","nonMaxSuppressionAsync"),u=ye(o,i,n,s,r,a);n=u.maxOutputSize,s=u.iouThreshold,r=u.scoreThreshold,a=u.softNmsSigma;const c=await Promise.all([o.data(),i.data()]),p=c[0],h=c[1],{selectedIndices:m,selectedScores:y}=Xg(p,h,n,s,r,a);return o!==e&&o.dispose(),i!==t&&i.dispose(),{selectedIndices:dt(m,"int32"),selectedScores:dt(y)}}const ry=sy;/**
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
 */function ay(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,a=!1){const o=f(e,"boxes","nonMaxSuppression"),i=f(t,"scores","nonMaxSuppression"),u=ye(o,i,n,s,r,null),c=u.maxOutputSize,p=u.iouThreshold,h=u.scoreThreshold,m={boxes:o,scores:i},y={maxOutputSize:c,iouThreshold:p,scoreThreshold:h,padToMaxOutputSize:a},T=b.runKernel(Mi,m,y);return{selectedIndices:T[0],validOutputs:T[1]}}const oy=g({nonMaxSuppressionPadded_:ay});/**
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
 */async function iy(e,t,n,s=.5,r=Number.NEGATIVE_INFINITY,a=!1){const o=f(e,"boxes","nonMaxSuppressionAsync"),i=f(t,"scores","nonMaxSuppressionAsync"),u=ye(o,i,n,s,r,null),c=u.maxOutputSize,p=u.iouThreshold,h=u.scoreThreshold,[m,y]=await Promise.all([o.data(),i.data()]),{selectedIndices:T,validOutputs:N}=Yg(m,y,c,p,h,a);return o!==e&&o.dispose(),i!==t&&i.dispose(),{selectedIndices:dt(T,"int32"),validOutputs:B(N,"int32")}}const uy=iy;/**
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
 */function cy(e,t,n=!1,s=!1){const r=f(e,"images","resizeBilinear");d(r.rank===3||r.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${r.rank}.`),d(t.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),d(s===!1||n===!1,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let a=r,o=!1;r.rank===3&&(o=!0,a=S(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const i={images:a},u={alignCorners:n,halfPixelCenters:s,size:t},c=b.runKernel(pu,i,u);return o?S(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const ly=g({resizeBilinear_:cy});/**
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
 */function py(e,t,n=!1,s=!1){const r=f(e,"images","resizeNearestNeighbor");d(r.rank===3||r.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${r.rank}.`),d(t.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),d(r.dtype==="float32"||r.dtype==="int32",()=>"`images` must have `int32` or `float32` as dtype"),d(s===!1||n===!1,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let a=r,o=!1;r.rank===3&&(o=!0,a=S(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const i={images:a},u={alignCorners:n,halfPixelCenters:s,size:t},c=b.runKernel(lu,i,u);return o?S(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const hy=g({resizeNearestNeighbor_:py});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fy(e,t="binary",n=!1,s=.5){const r=f(e,"image","threshold"),a=.2989,o=.587,i=.114,u=r.shape[0]*r.shape[1];let c=D(dt([s]),255),p,h,m,y;if(d(r.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${r.rank}.`),d(r.shape[2]===3||r.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${r.shape[2]}.`),d(r.dtype==="int32"||r.dtype==="float32",()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${r.dtype}.`),d(t==="otsu"||t==="binary",()=>`Method must be binary or otsu, but was ${t}`),r.shape[2]===3){[p,h,m]=Ae(r,[1,1,1],-1);const w=D(p,a),k=D(h,o),v=D(m,i);y=Y(Y(w,k),v)}else y=e;if(t==="otsu"){const w=ta(H($a(y),"int32"),Nt([]),256);c=my(w,u)}const T=n?ws(y,c):mn(y,c);return H(D(T,255),"int32")}function my(e,t){let n=dt([-1]),s=dt([0]),r=dt([0]),a,o,i,u,c,p;for(let h=0;h<e.size-1;h++){a=R(e,0,h+1),o=R(e,h+1),c=st(j(a),t),p=st(j(o),t);const m=j(D(a,De(0,a.size)));i=st(m,j(a));const y=cn(o.shape,a.size),T=Y(De(0,o.size),y),N=D(o,T);u=st(j(N),j(o));const w=O(i,u),k=O(i,u),v=D(c,p);r=D(D(v,w),k);const $=mn(r,s);s=de($,r,s),n=de($,dt([h]),n)}return n}const dy=g({threshold_:fy});/**
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
 */function gy(e,t,n="nearest",s="constant",r=0,a){const o=f(e,"image","transform","float32"),i=f(t,"transforms","transform","float32");d(o.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${o.rank}.`),d(i.rank===2&&(i.shape[0]===o.shape[0]||i.shape[0]===1)&&i.shape[1]===8,()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),d(a==null||a.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${a}.`);const u={image:o,transforms:i},c={interpolation:n,fillMode:s,fillValue:r,outputShape:a};return b.runKernel(qu,u,c)}const yy=g({transform_:gy});/**
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
 */function by(e,t,n){d(t%1===0,()=>`bandPart(): numLower must be an integer, got ${t}.`),d(n%1===0,()=>`bandPart(): numUpper must be an integer, got ${n}.`);const s=f(e,"a","bandPart");d(s.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${s.rank}.`);const r=s.shape,[a,o]=s.shape.slice(-2);if(!(t<=a))throw new Error(`bandPart(): numLower (${t}) must not be greater than the number of rows (${a}).`);if(!(n<=o))throw new Error(`bandPart(): numUpper (${n}) must not be greater than the number of columns (${o}).`);t<0&&(t=a),n<0&&(n=o);const i=S(De(0,a,1,"int32"),[-1,1]),u=De(0,o,1,"int32"),c=O(i,u),p=Xe(ws(c,B(+t,"int32")),ca(c,B(-n,"int32"))),h=ge([a,o],s.dtype);return S(It(ne(S(s,[-1,a,o])).map(m=>de(p,m,h))),r)}const wy=g({bandPart_:by});/**
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
 */function Ny(e){let t;if(Array.isArray(e)){t=!1,d(e!=null&&e.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const r=e[0].shape[0];for(let a=1;a<e.length;++a)d(e[a].shape[0]===r,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${e[a].shape[0]} vs. ${r})`)}else t=!0,e=Ae(e,e.shape[0],0).map(r=>$s(r,[0]));d(e.length<=e[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${e.length}) exceeds number of dimensions (${e[0].shape[0]}).`);const n=[],s=e;for(let r=0;r<e.length;++r)n.push(b.tidy(()=>{let a=s[r];if(r>0)for(let o=0;o<r;++o){const i=D(j(D(n[o],a)),n[o]);a=O(a,i)}return st(a,fn(a,"euclidean"))}));return t?It(n,0):n}const Ty=g({gramSchmidt_:Ny});/**
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
 */function Sy(e,t=!1){if(d(e.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${e.rank}`),e.rank===2)return Ys(e,t);{const n=e.shape.slice(0,e.shape.length-2).reduce((u,c)=>u*c),s=ne(S(e,[n,e.shape[e.shape.length-2],e.shape[e.shape.length-1]]),0),r=[],a=[];s.forEach(u=>{const[c,p]=Ys(u,t);r.push(c),a.push(p)});const o=S(It(r,0),e.shape),i=S(It(a,0),e.shape);return[o,i]}}function Ys(e,t=!1){return b.tidy(()=>{d(e.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${e.shape.length}D Tensor.`);const n=e.shape[0],s=e.shape[1];let r=oa(n),a=Ct(e);const o=$e([[1]],[1,1]);let i=Ct(o);const u=n>=s?s:n;for(let c=0;c<u;++c){const p=a,h=i,m=r;[i,a,r]=b.tidy(()=>{const y=R(a,[c,c],[n-c,1]),T=fn(y),N=R(a,[c,c],[1,1]),w=de(mn(N,0),$e([[-1]]),$e([[1]])),k=O(N,D(w,T)),v=st(y,k);v.shape[0]===1?i=Ct(o):i=rt([o,R(v,[1,0],[v.shape[0]-1,v.shape[1]])],0);const $=$t(st(P(w,k),T)),E=R(a,[c,0],[n-c,s]),I=D($,i),_=Vn(i);if(c===0)a=O(E,P(I,P(_,E)));else{const C=O(E,P(I,P(_,E)));a=rt([R(a,[0,0],[c,s]),C],0)}const A=Vn(I),F=R(r,[0,c],[n,r.shape[1]-c]);if(c===0)r=O(F,P(P(F,i),A));else{const C=O(F,P(P(F,i),A));r=rt([R(r,[0,0],[n,c]),C],1)}return[i,a,r]}),$l([p,h,m])}return!t&&n>s&&(r=R(r,[0,0],[n,s]),a=R(a,[0,0],[s,s])),[r,a]})}const ky=g({qr_:Sy});/**
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
 */var ot;(function(e){e[e.NONE=0]="NONE",e[e.MEAN=1]="MEAN",e[e.SUM=2]="SUM",e[e.SUM_BY_NONZERO_WEIGHTS=3]="SUM_BY_NONZERO_WEIGHTS"})(ot||(ot={}));function $y(e,t,n=ot.SUM_BY_NONZERO_WEIGHTS){const s=f(e,"losses","computeWeightedLoss");let r=null;t!=null&&(r=f(t,"weights","computeWeightedLoss"));const a=r==null?s:D(s,r);if(n===ot.NONE)return a;if(n===ot.SUM)return j(a);if(n===ot.MEAN){if(r==null)return Je(a);{const o=s.size/r.size,i=st(j(a),j(r));return o>1?st(i,B(o)):i}}if(n===ot.SUM_BY_NONZERO_WEIGHTS){if(r==null)return st(j(a),B(s.size));{const o=D(r,qt(s.shape)),i=H(j(ba(o,B(0))),"float32");return st(j(a),i)}}throw Error(`Unknown reduction: ${n}`)}const xt=g({computeWeightedLoss_:$y});/**
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
 */function Ey(e,t,n,s=ot.SUM_BY_NONZERO_WEIGHTS){const r=f(e,"labels","absoluteDifference"),a=f(t,"predictions","absoluteDifference");let o=null;n!=null&&(o=f(n,"weights","absoluteDifference")),at(r.shape,a.shape,"Error in absoluteDifference: ");const i=ht(O(r,a));return xt(i,o,s)}const _y=g({absoluteDifference_:Ey});function vy(e,t,n,s,r=ot.SUM_BY_NONZERO_WEIGHTS){const a=f(e,"labels","cosineDistance"),o=f(t,"predictions","cosineDistance");let i=null;s!=null&&(i=f(s,"weights","cosineDistance")),at(a.shape,o.shape,"Error in cosineDistance: ");const u=B(1),c=O(u,j(D(a,o),n,!0));return xt(c,i,r)}const Iy=g({cosineDistance_:vy});function xy(e,t,n,s=ot.SUM_BY_NONZERO_WEIGHTS){let r=f(e,"labels","hingeLoss");const a=f(t,"predictions","hingeLoss");let o=null;n!=null&&(o=f(n,"weights","hingeLoss")),at(r.shape,a.shape,"Error in hingeLoss: ");const i=B(1);r=O(D(B(2),r),i);const u=gn(O(i,D(r,a)));return xt(u,o,s)}const Dy=g({hingeLoss_:xy});/**
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
 */function Ay(e,t,n,s=1,r=ot.SUM_BY_NONZERO_WEIGHTS){const a=f(e,"labels","huberLoss"),o=f(t,"predictions","huberLoss");let i=null;n!=null&&(i=f(n,"weights","huberLoss")),at(a.shape,o.shape,"Error in huberLoss: ");const u=B(s),c=ht(O(o,a)),p=ya(c,u),h=O(c,p),m=Y(D(B(.5),hn(p)),D(u,h));return xt(m,i,r)}const Oy=g({huberLoss_:Ay});/**
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
 */function Cy(e,t,n,s=1e-7,r=ot.SUM_BY_NONZERO_WEIGHTS){const a=f(e,"labels","logLoss"),o=f(t,"predictions","logLoss");let i=null;n!=null&&(i=f(n,"weights","logLoss")),at(a.shape,o.shape,"Error in logLoss: ");const u=B(1),c=B(s),p=$t(D(a,xe(Y(o,c)))),h=D(O(u,a),xe(Y(O(u,o),c))),m=O(p,h);return xt(m,i,r)}const Fy=g({logLoss_:Cy});/**
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
 */function By(e,t,n,s=ot.SUM_BY_NONZERO_WEIGHTS){const r=f(e,"labels","meanSquaredError"),a=f(t,"predictions","meanSquaredError");let o=null;n!=null&&(o=f(n,"weights","meanSquaredError")),at(r.shape,a.shape,"Error in meanSquaredError: ");const i=_a(r,a);return xt(i,o,s)}const Py=g({meanSquaredError_:By});/**
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
 */function Ly(e,t){const n=f(e,"labels","sigmoidCrossEntropyWithLogits"),s=f(t,"logits","sigmoidCrossEntropyWithLogits");at(n.shape,s.shape,"Error in sigmoidCrossEntropyWithLogits: ");const r=gn(s),a=D(s,n),o=pa(Zt($t(ht(s))));return Y(O(r,a),o)}function Ry(e,t,n,s=0,r=ot.SUM_BY_NONZERO_WEIGHTS){let a=f(e,"multiClassLabels","sigmoidCrossEntropy");const o=f(t,"logits","sigmoidCrossEntropy");let i=null;if(n!=null&&(i=f(n,"weights","sigmoidCrossEntropy")),at(a.shape,o.shape,"Error in sigmoidCrossEntropy: "),s>0){const c=B(s),p=B(1),h=B(.5);a=Y(D(a,O(p,c)),D(h,c))}const u=Ly(a,o);return xt(u,i,r)}const zy=g({sigmoidCrossEntropy_:Ry});/**
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
 */function Vy(e,t,n=-1){if(n===-1&&(n=t.rank-1),n!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${n}`);return vt((r,a,o)=>{const u=fa(a,[n],!0),c=O(H(a,"float32"),u);o([r,c]);const p=$t(D(c,r));return{value:j(p,[n]),gradFunc:(y,T)=>{const[N,w]=T,k=pn(y.shape,[n]);return[D(S(y,k),O(H(N,"float32"),Zt(w))),D(S(y,k),O(Zt(w),H(N,"float32")))]}}})(e,t)}function jy(e,t,n,s=0,r=ot.SUM_BY_NONZERO_WEIGHTS){let a=f(e,"onehotLabels","softmaxCrossEntropy");const o=f(t,"logits","softmaxCrossEntropy");let i=null;if(n!=null&&(i=f(n,"weights","softmaxCrossEntropy")),at(a.shape,o.shape,"Error in softmaxCrossEntropy: "),s>0){const c=B(s),p=B(1),h=B(a.shape[1]);a=Y(D(a,O(p,c)),st(c,h))}const u=Vy(a,o);return xt(u,i,r)}const Ky=g({softmaxCrossEntropy_:jy});/**
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
 */function Uy(e,t,n,s){const r=f(e,"indices","sparseFillEmptyRows","int32"),a=f(t,"values","sparseFillEmptyRows"),o=f(n,"denseShape","sparseFillEmptyRows","int32"),i=f(s,"defaultValue","sparseFillEmptyRows",a.dtype);if(r.rank!==2)throw new Error(`Indices should be Tensor2D but received shape
        ${r.shape}`);if(a.rank!==1)throw new Error(`Values should be Tensor1D but received shape ${a.shape}`);if(o.rank!==1)throw new Error(`Dense shape should be Tensor1D but received shape ${o.shape}`);if(i.rank!==0)throw new Error(`Default value should be a scalar but received shape ${i.shape}`);const u={indices:r,values:a,denseShape:o,defaultValue:i},c=b.runKernel(Au,u);return{outputIndices:c[0],outputValues:c[1],emptyRowIndicator:c[2],reverseIndexMap:c[3]}}const Wy=g({sparseFillEmptyRows_:Uy});/**
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
 */function qy(e,t,n){const s=f(e,"inputIndices","sparseReshape","int32"),r=f(t,"inputShape","sparseReshape","int32"),a=f(n,"newShape","sparseReshape","int32");if(s.rank!==2)throw new Error(`Input indices should be Tensor2D but received shape
        ${s.shape}`);if(r.rank!==1)throw new Error(`Input shape should be Tensor1D but received shape ${r.shape}`);if(a.rank!==1)throw new Error(`New shape should be Tensor1D but received shape ${a.shape}`);const o={inputIndices:s,inputShape:r,newShape:a},i=b.runKernel(Ou,o);return{outputIndices:i[0],outputShape:i[1]}}const Hy=g({sparseReshape_:qy});/**
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
 */function Gy(e,t,n){const s=f(e,"data","sparseSegmentMean"),r=f(t,"indices","sparseSegmentMean","int32"),a=f(n,"segmentIds","sparseSegmentMean","int32");if(s.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
          ${r.shape}`);if(a.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
          ${a.shape}`);const o={data:s,indices:r,segmentIds:a};return b.runKernel(Cu,o)}const My=g({sparseSegmentMean_:Gy});/**
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
 */function Yy(e,t,n){const s=f(e,"data","sparseSegmentSum"),r=f(t,"indices","sparseSegmentSum","int32"),a=f(n,"segmentIds","sparseSegmentSum","int32");if(s.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
         ${r.shape}`);if(a.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
         ${a.shape}`);const o={data:s,indices:r,segmentIds:a};return b.runKernel(Fu,o)}const Xy=g({sparseSegmentSum_:Yy});/**
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
 */function Jy(e,t,n,s,r,a,o,i){const u=f(e,"data","stringNGrams","string");if(u.dtype!=="string")throw new Error("Data must be of datatype string");if(u.shape.length!==1)throw new Error(`Data must be a vector, saw: ${u.shape}`);const c=f(t,"dataSplits","stringNGrams");if(c.dtype!=="int32")throw new Error("Data splits must be of datatype int32");const p={separator:n,nGramWidths:s,leftPad:r,rightPad:a,padWidth:o,preserveShortSequences:i},h={data:u,dataSplits:c},m=b.runKernel(Ru,h,p);return{nGrams:m[0],nGramsSplits:m[1]}}const Zy=g({stringNGrams_:Jy});/**
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
 */function Qy(e,t,n=!0){const s=f(e,"input","stringSplit","string"),r=f(t,"delimiter","stringSplit","string");if(s.rank!==1)throw new Error(`Input should be Tensor1D but received shape ${s.shape}`);if(r.rank!==0)throw new Error(`Delimiter should be a scalar but received shape ${r.shape}`);const a={skipEmpty:n},o={input:s,delimiter:r},i=b.runKernel(zu,o,a);return{indices:i[0],values:i[1],shape:i[2]}}const tb=g({stringSplit_:Qy});/**
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
 */function eb(e,t){const n=f(e,"input","stringToHashBucketFast","string"),s={numBuckets:t};if(t<=0)throw new Error("Number of buckets must be at least 1");const r={input:n};return b.runKernel(Vu,r,s)}const nb=g({stringToHashBucketFast_:eb});/**
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
 */const sb={fft:Ss,ifft:Ze,rfft:ks,irfft:Ea},rb={hammingWindow:xg,hannWindow:Da,frame:Aa,stft:Cg},ab={flipLeftRight:Lg,grayscaleToRGB:zg,resizeNearestNeighbor:hy,resizeBilinear:ly,rotateWithOffset:jg,cropAndResize:Bg,nonMaxSuppression:Ug,nonMaxSuppressionAsync:ty,nonMaxSuppressionWithScore:ny,nonMaxSuppressionWithScoreAsync:ry,nonMaxSuppressionPadded:oy,nonMaxSuppressionPaddedAsync:uy,threshold:dy,transform:yy},ob={bandPart:wy,gramSchmidt:Ty,qr:ky},ib={absoluteDifference:_y,computeWeightedLoss:xt,cosineDistance:Iy,hingeLoss:Dy,huberLoss:Oy,logLoss:Fy,meanSquaredError:Py,sigmoidCrossEntropy:zy,softmaxCrossEntropy:Ky},ub={sparseFillEmptyRows:Wy,sparseReshape:Hy,sparseSegmentMean:My,sparseSegmentSum:Xy},cb={stringNGrams:Zy,stringSplit:tb,stringToHashBucketFast:nb};/**
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
 */const lb=x();lb.registerFlag("KEEP_INTERMEDIATE_TENSORS",()=>!1,e=>{e&&console.warn("Keep intermediate tensors is ON. This will print the values of all intermediate tensors during model inference. Not all models support this mode. For details, check e2e/benchmarks/ model_config.js. This significantly impacts performance.")});/**
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
 *
 * =============================================================================
 */var pt;(function(e){e[e.DT_INVALID=0]="DT_INVALID",e[e.DT_FLOAT=1]="DT_FLOAT",e[e.DT_DOUBLE=2]="DT_DOUBLE",e[e.DT_INT32=3]="DT_INT32",e[e.DT_UINT8=4]="DT_UINT8",e[e.DT_INT16=5]="DT_INT16",e[e.DT_INT8=6]="DT_INT8",e[e.DT_STRING=7]="DT_STRING",e[e.DT_COMPLEX64=8]="DT_COMPLEX64",e[e.DT_INT64=9]="DT_INT64",e[e.DT_BOOL=10]="DT_BOOL",e[e.DT_QINT8=11]="DT_QINT8",e[e.DT_QUINT8=12]="DT_QUINT8",e[e.DT_QINT32=13]="DT_QINT32",e[e.DT_BFLOAT16=14]="DT_BFLOAT16",e[e.DT_QINT16=15]="DT_QINT16",e[e.DT_QUINT16=16]="DT_QUINT16",e[e.DT_UINT16=17]="DT_UINT16",e[e.DT_COMPLEX128=18]="DT_COMPLEX128",e[e.DT_HALF=19]="DT_HALF",e[e.DT_RESOURCE=20]="DT_RESOURCE",e[e.DT_VARIANT=21]="DT_VARIANT",e[e.DT_UINT32=22]="DT_UINT32",e[e.DT_UINT64=23]="DT_UINT64",e[e.DT_FLOAT_REF=101]="DT_FLOAT_REF",e[e.DT_DOUBLE_REF=102]="DT_DOUBLE_REF",e[e.DT_INT32_REF=103]="DT_INT32_REF",e[e.DT_UINT8_REF=104]="DT_UINT8_REF",e[e.DT_INT16_REF=105]="DT_INT16_REF",e[e.DT_INT8_REF=106]="DT_INT8_REF",e[e.DT_STRING_REF=107]="DT_STRING_REF",e[e.DT_COMPLEX64_REF=108]="DT_COMPLEX64_REF",e[e.DT_INT64_REF=109]="DT_INT64_REF",e[e.DT_BOOL_REF=110]="DT_BOOL_REF",e[e.DT_QINT8_REF=111]="DT_QINT8_REF",e[e.DT_QUINT8_REF=112]="DT_QUINT8_REF",e[e.DT_QINT32_REF=113]="DT_QINT32_REF",e[e.DT_BFLOAT16_REF=114]="DT_BFLOAT16_REF",e[e.DT_QINT16_REF=115]="DT_QINT16_REF",e[e.DT_QUINT16_REF=116]="DT_QUINT16_REF",e[e.DT_UINT16_REF=117]="DT_UINT16_REF",e[e.DT_COMPLEX128_REF=118]="DT_COMPLEX128_REF",e[e.DT_HALF_REF=119]="DT_HALF_REF",e[e.DT_RESOURCE_REF=120]="DT_RESOURCE_REF",e[e.DT_VARIANT_REF=121]="DT_VARIANT_REF",e[e.DT_UINT32_REF=122]="DT_UINT32_REF",e[e.DT_UINT64_REF=123]="DT_UINT64_REF"})(pt||(pt={}));var Xs;(function(e){(function(t){t[t.LEGACY=0]="LEGACY",t[t.V1=1]="V1",t[t.V2=2]="V2"})(e.CheckpointFormatVersion||(e.CheckpointFormatVersion={}))})(Xs||(Xs={}));/**
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
 */const As={};function jN(e,t){const n={tfOpName:e,category:"custom",inputs:[],attrs:[],customExecutor:t};As[e]=n}function Oa(e){return As[e]}function KN(e){delete As[e]}/**
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
 */function l(e,t,n,s,r){const a=t.inputParams[e];if(a&&a.inputIndexStart!==void 0){const i=a.inputIndexStart,u=a.inputIndexEnd===0?void 0:a.inputIndexEnd===void 0?i+1:a.inputIndexEnd;if(a.type==="tensor")return nt(t.inputNames[a.inputIndexStart],n,s,r);if(a.type==="tensors")return t.inputNames.slice(i,u).map(m=>nt(m,n,s,r));const c=nt(t.inputNames.slice(i)[0],n,s,r),p=c.dataSync();return a.type==="number"?p[0]:Ht(c.shape,p)}const o=t.attrParams[e];return o&&o.value}function nt(e,t,n,s){const[r,a]=ut(e);if(s!=null){const i=s.getHashTableHandleByName(r);if(i!=null)return i}const o=n.currentContextIds.find(i=>!!t[Qe(r,i)]);return o!==void 0?t[Qe(r,o)][a]:void 0}function pb(e,t,n){return t[Qe(e,n.currentContextId)]}function wt(e,t){const[n,s,r]=ut(e);return[Qe(n,t&&t.currentContextId),s,r]}function Qe(e,t){return t?`${e}-${t}`:e}function ut(e){const t=e.split(":");if(t.length===1)return[e,0,void 0];const n=t[0],s=t.length===3?t[1]:void 0,r=Number(t[t.length-1]);return[n,r,s]}function Ue(e,t,n){let s=l("pad",e,t,n);if(s==="explicit"){s=l("explicitPaddings",e,t,n);const r=[[0,0],[0,0],[0,0],[0,0]];for(let a=0;a<4;a++)r[a][0]=s[a*2],r[a][1]=s[a*2+1];return r}return s}function St(e){return e.kept?e:Ct(e)}/**
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
 */const hb=[{tfOpName:"Add",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddV2",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddN",category:"arithmetic",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"BiasAdd",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"Sub",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"RealDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Div",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"DivNoNan",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mul",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Maximum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Minimum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Pow",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SquaredDifference",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorMod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],fb=Object.freeze(Object.defineProperty({__proto__:null,json:hb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const mb=[{tfOpName:"Abs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan2",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Ceil",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ClipByValue",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"clipValueMin",type:"number"},{start:2,name:"clipValueMax",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Complex",category:"basic_math",inputs:[{start:0,name:"real",type:"tensor"},{start:1,name:"imag",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ComplexAbs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Elu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Exp",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Floor",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Imag",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Neg",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Real",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Prelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"alpha",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu6",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Selu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sigmoid",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Rsqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Square",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sign",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Round",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Expm1",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log1p",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Reciprocal",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Softplus",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Erf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Prod",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axes",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LeakyRelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"alpha",name:"alpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsNan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],db=Object.freeze(Object.defineProperty({__proto__:null,json:mb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const gb=[{tfOpName:"EmptyTensorList",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"maxNumElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"LoopCond",category:"control",inputs:[{start:0,name:"pred",type:"tensor"}]},{tfOpName:"Switch",category:"control",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"pred",type:"tensor"}]},{tfOpName:"Merge",category:"control",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"Enter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"frame_name",name:"frameName",type:"string"},{tfName:"is_constant",name:"isConstant",type:"bool"}]},{tfOpName:"Exit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NextIteration",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayV3",category:"control",inputs:[{start:0,name:"size",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"dynamic_size",name:"dynamicSize",type:"bool"},{tfName:"clear_after_read",name:"clearAfterRead",type:"bool"},{tfName:"identical_element_shapes",name:"identicalElementShapes",type:"bool"},{tfName:"tensor_array_name",name:"name",type:"string"}]},{tfOpName:"TensorArrayWriteV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayReadV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayGatherV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"}]},{tfOpName:"TensorArrayScatterV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArrayConcatV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape_except0",name:"elementShapeExcept0",type:"shape",notSupported:!0}]},{tfOpName:"TensorArraySplitV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"tensor",type:"tensor"},{start:2,name:"lengths",type:"number[]"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArraySizeV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}]},{tfOpName:"TensorArrayCloseV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"}]},{tfOpName:"StatelessIf",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"If",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"StatelessWhile",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"While",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"TensorListScatter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListScatterV2",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"},{start:3,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGather",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListSetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListReserve",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListFromTensor",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListStack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"},{tfName:"num_elements",name:"numElements",type:"dtype"}]},{tfOpName:"TensorListSplit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"},{start:2,name:"lengths",type:"number[]"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcat",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcatV2",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPopBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPushBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListLength",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}]},{tfOpName:"TensorListResize",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"size",type:"number"}]}],yb=Object.freeze(Object.defineProperty({__proto__:null,json:gb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const bb=[{tfOpName:"AvgPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[],notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPoolWithArgmax",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"include_batch_in_index",name:"includeBatchInIndex",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AvgPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Conv1D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"stride",name:"stride",type:"number"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NWC"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"dilation",name:"dilation",type:"number",defaultValue:1}]},{tfOpName:"Conv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"useCudnnOnGpu",name:"useCudnnOnGpu",type:"bool"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"_FusedConv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"use_cudnn_on_gpu",name:"useCudnnOnGpu",type:"bool",defaultValue:!0},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2}]},{tfOpName:"Conv2DBackpropInput",category:"convolution",inputs:[{start:2,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:0,name:"outputShape",type:"number[]"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]",notSupported:!0}]},{tfOpName:"DepthwiseConv2d",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"DepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"FusedDepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]}]},{tfOpName:"Conv3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"Dilation2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"rates",name:"dilations",type:"number[]"},{tfName:"padding",name:"pad",type:"string"}]}],wb=Object.freeze(Object.defineProperty({__proto__:null,json:bb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Nb=[{tfOpName:"Fill",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"},{start:1,name:"value",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"LinSpace",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"num",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"OneHot",category:"creation",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"depth",type:"number"},{start:2,name:"onValue",type:"number",defaultValue:1},{start:3,name:"offValue",type:"number",defaultValue:0}],attrs:[{tfName:"axis",name:"axis",type:"number",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Ones",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"OnesLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"RandomStandardNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniform",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number",defaultValue:0},{tfName:"maxval",name:"maxval",type:"number",defaultValue:1},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Range",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"step",type:"number",defaultValue:0}],attrs:[{tfName:"Tidx",name:"dtype",type:"dtype"}]},{tfOpName:"TruncatedNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"means",name:"mean",type:"number",defaultValue:0},{tfName:"stddev",name:"stdDev",type:"number",defaultValue:1},{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Zeros",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"ZerosLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Multinomial",category:"creation",inputs:[{start:0,name:"logits",type:"tensor"},{start:1,name:"numSamples",type:"number"}],attrs:[{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number"},{tfName:"T",name:"dtype",type:"dtype"},{tfName:"output_dtype",name:"output_dtype",type:"dtype"}]}],Tb=Object.freeze(Object.defineProperty({__proto__:null,json:Nb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Sb=[{tfOpName:"NonMaxSuppressionV2",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV3",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV4",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"T_threshold",name:"threshold",type:"dtype",notSupported:!0},{tfName:"pad_to_max_output_size",name:"padToMaxOutputSize",type:"bool"}]},{tfOpName:"NonMaxSuppressionV5",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"},{start:5,name:"softNmsSigma",type:"number"}]},{tfOpName:"Where",category:"dynamic",inputs:[{start:0,name:"condition",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ListDiff",category:"dynamic",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],kb=Object.freeze(Object.defineProperty({__proto__:null,json:Sb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const $b=[{tfOpName:"LowerBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"TopKV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"k",type:"number"}],attrs:[{tfName:"sorted",name:"sorted",type:"bool"}]},{tfOpName:"UpperBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"Unique",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"UniqueV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]}],Eb=Object.freeze(Object.defineProperty({__proto__:null,json:$b},Symbol.toStringTag,{value:"Module"}));/**
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
 */const _b=[{tfOpName:"PlaceholderWithDefault",category:"graph",inputs:[{start:0,name:"default",type:"tensor"}],attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Placeholder",category:"graph",attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Const",category:"graph"},{tfOpName:"Identity",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IdentityN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Snapshot",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Rank",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Size",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Shape",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"ShapeN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Print",category:"graph",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"data",type:"tensors"}],attrs:[{tfName:"message",name:"message",type:"string"},{tfName:"first_n",name:"firstN",type:"number",notSupported:!0},{tfName:"summarize",name:"summarize",type:"number",defaultValue:3}]},{tfOpName:"NoOp",category:"graph",inputs:[]},{tfOpName:"StopGradient",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"FakeQuantWithMinMaxVars",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"min",name:"min",type:"number"},{tfName:"max",name:"max",type:"number"}]}],vb=Object.freeze(Object.defineProperty({__proto__:null,json:_b},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Ib=[{tfOpName:"HashTable",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"HashTableV2",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"LookupTableImport",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableImportV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFind",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFindV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableSize",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"LookupTableSizeV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]}],xb=Object.freeze(Object.defineProperty({__proto__:null,json:Ib},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Db=[{tfOpName:"ResizeBilinear",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ResizeNearestNeighbor",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"CropAndResize",category:"image",inputs:[{start:0,name:"image",type:"tensor"},{start:1,name:"boxes",type:"tensor"},{start:2,name:"boxInd",type:"tensor"},{start:3,name:"cropSize",type:"number[]"}],attrs:[{tfName:"method",name:"method",type:"string"},{tfName:"extrapolation_value",name:"extrapolationValue",type:"number"}]},{tfOpName:"ImageProjectiveTransformV3",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"transforms",type:"tensor"},{start:2,name:"outputShape",type:"number[]"},{start:3,name:"fillValue",type:"number"}],attrs:[{tfName:"interpolation",name:"interpolation",type:"string"},{tfName:"fill_mode",name:"fillMode",type:"string"}]}],Ab=Object.freeze(Object.defineProperty({__proto__:null,json:Db},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Ob=[{tfOpName:"Equal",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NotEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Greater",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"GreaterEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Less",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LessEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalAnd",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalNot",category:"logical",inputs:[{start:0,name:"a",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalOr",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Select",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SelectV2",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],Cb=Object.freeze(Object.defineProperty({__proto__:null,json:Ob},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Fb=[{tfOpName:"_FusedMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMulV2",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Transpose",category:"matrices",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"perm",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Einsum",category:"matrices",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"equation",name:"equation",type:"string"},{tfName:"N",name:"n",type:"number",defaultValue:2},{tfName:"T",name:"dtype",type:"dtype"}]}],Bb=Object.freeze(Object.defineProperty({__proto__:null,json:Fb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Pb=[{tfOpName:"EuclideanNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",defaultValue:!1}]},{tfOpName:"FusedBatchNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV2",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV3",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"LRN",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"depth_radius",name:"radius",type:"number",defaultValue:5},{tfName:"bias",name:"bias",type:"number",defaultValue:1},{tfName:"alpha",name:"alpha",type:"number",defaultValue:1},{tfName:"beta",name:"beta",type:"number",defaultValue:.5}]},{tfOpName:"Softmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"LogSoftmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"SparseToDense",category:"normalization",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!0,notSupported:!0}]}],Lb=Object.freeze(Object.defineProperty({__proto__:null,json:Pb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Rb=[{tfOpName:"Bincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}]},{tfOpName:"DenseBincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}],attrs:[{tfName:"binary_output",name:"binaryOutput",type:"bool"}]},{tfOpName:"Max",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Mean",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Min",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Sum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"All",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Any",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"ArgMax",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"ArgMin",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"Prod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Cumprod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]},{tfOpName:"Cumsum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]}],zb=Object.freeze(Object.defineProperty({__proto__:null,json:Rb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Vb=[{tfOpName:"ConcatV2",category:"slice_join",inputs:[{start:0,end:-1,name:"tensors",type:"tensors"},{start:-1,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"Concat",category:"slice_join",inputs:[{start:1,end:0,name:"tensors",type:"tensors"},{start:0,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"GatherV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"axis",type:"number",defaultValue:0}],attrs:[{tfName:"batch_dims",name:"batchDims",type:"number",defaultValue:0}]},{tfOpName:"Gather",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",notSupported:!0}]},{tfOpName:"Reverse",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"dims",type:"bool[]"}]},{tfOpName:"ReverseV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}]},{tfOpName:"Slice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"size",type:"number[]"}]},{tfOpName:"StridedSlice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"end",type:"number[]"},{start:3,name:"strides",type:"number[]"}],attrs:[{tfName:"begin_mask",name:"beginMask",type:"number",defaultValue:0},{tfName:"end_mask",name:"endMask",type:"number",defaultValue:0},{tfName:"new_axis_mask",name:"newAxisMask",type:"number",defaultValue:0},{tfName:"ellipsis_mask",name:"ellipsisMask",type:"number",defaultValue:0},{tfName:"shrink_axis_mask",name:"shrinkAxisMask",type:"number",defaultValue:0}]},{tfOpName:"Pack",category:"slice_join",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0}]},{tfOpName:"Unpack",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0},{tfName:"num",name:"num",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Tile",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"reps",type:"number[]"}]},{tfOpName:"Split",category:"slice_join",inputs:[{start:0,name:"axis",type:"number",defaultValue:0},{start:1,name:"x",type:"tensor"}],attrs:[{tfName:"num_split",name:"numOrSizeSplits",type:"number",defaultValue:1}]},{tfOpName:"SplitV",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"numOrSizeSplits",type:"number[]"},{start:2,name:"axis",type:"number",defaultValue:0}]},{tfOpName:"ScatterNd",category:"slice_join",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"shape",type:"number[]"}]},{tfOpName:"GatherNd",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}]},{tfOpName:"SparseToDense",category:"slice_join",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!1,notSupported:!0}]}],jb=Object.freeze(Object.defineProperty({__proto__:null,json:Vb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Kb=[{tfOpName:"SparseFillEmptyRows",category:"sparse",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"denseShape",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}]},{tfOpName:"SparseReshape",category:"sparse",inputs:[{start:0,name:"inputIndices",type:"tensor"},{start:1,name:"inputShape",type:"tensor"},{start:2,name:"newShape",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SparseSegmentMean",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]},{tfOpName:"SparseSegmentSum",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]}],Ub=Object.freeze(Object.defineProperty({__proto__:null,json:Kb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Wb=[{tfOpName:"FFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"RFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]},{tfOpName:"IRFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]}],qb=Object.freeze(Object.defineProperty({__proto__:null,json:Wb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Hb=[{tfOpName:"StringNGrams",category:"string",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"dataSplits",type:"tensor"}],attrs:[{tfName:"separator",name:"separator",type:"string"},{tfName:"ngram_widths",name:"nGramWidths",type:"number[]"},{tfName:"left_pad",name:"leftPad",type:"string"},{tfName:"right_pad",name:"rightPad",type:"string"},{tfName:"pad_width",name:"padWidth",type:"number"},{tfName:"preserve_short_sequences",name:"preserveShortSequences",type:"bool"}],outputs:["ngrams","ngrams_splits"]},{tfOpName:"StringSplit",category:"string",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"delimiter",type:"tensor"}],attrs:[{tfName:"skip_empty",name:"skipEmpty",type:"bool"}],outputs:["indices","values","shape"]},{tfOpName:"StringToHashBucketFast",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"num_buckets",name:"numBuckets",type:"number"}]}],Gb=Object.freeze(Object.defineProperty({__proto__:null,json:Hb},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Mb=[{tfOpName:"Cast",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"SrcT",name:"sdtype",type:"dtype",notSupported:!0},{tfName:"DstT",name:"dtype",type:"dtype"}]},{tfOpName:"ExpandDims",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"MirrorPad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"mode",name:"mode",type:"string"}]},{tfOpName:"Pad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"constant_value",name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"PadV2",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"},{start:2,name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"Reshape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"Squeeze",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"axis",tfDeprecatedName:"squeeze_dims",name:"axis",type:"number[]"}]},{tfOpName:"SpaceToBatchND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"paddings",type:"number[]"}]},{tfOpName:"BatchToSpaceND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"crops",type:"number[]"}]},{tfOpName:"DepthToSpace",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"block_size",name:"blockSize",type:"number"},{tfName:"data_format",name:"dataFormat",type:"string"}]},{tfOpName:"BroadcastTo",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}],attrs:[]},{tfOpName:"BroadcastArgs",category:"transformation",inputs:[{start:0,name:"s0",type:"tensor"},{start:1,name:"s1",type:"tensor"}],attrs:[]}],Yb=Object.freeze(Object.defineProperty({__proto__:null,json:Mb},Symbol.toStringTag,{value:"Module"}));/**
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
 */class Js{static get Instance(){return this._instance||(this._instance=new this)}constructor(){const t=[fb,db,yb,wb,Tb,kb,Eb,vb,xb,Ab,Cb,Bb,Lb,zb,jb,Ub,qb,Gb,Yb],n=[].concat(...t.map(s=>s.json));this.opMappers=n.reduce((s,r)=>(s[r.tfOpName]=r,s),{})}transformGraph(t,n={}){const s=t.node,r=[],a=[],o=[],i=s.reduce((N,w)=>(N[w.name]=this.mapNode(w),w.op.startsWith("Placeholder")?r.push(N[w.name]):w.op==="Const"?a.push(N[w.name]):(w.input==null||w.input.length===0)&&o.push(N[w.name]),N),{});let u=[];const c=[];let p={},h={};n!=null&&(p=this.mapSignatureEntries(n.inputs),h=this.mapSignatureEntries(n.outputs));const m=Object.keys(i);m.forEach(N=>{const w=i[N];w.inputNames.forEach((k,v)=>{const[$,,E]=wt(k),I=i[$];if(I.outputs!=null){const _=I.outputs.indexOf(E);if(_!==-1){const A=`${$}:${_}`;w.inputNames[v]=A}}w.inputs.push(I),I.children.push(w)})}),Object.keys(h).length===0?m.forEach(N=>{const w=i[N];w.children.length===0&&c.push(w)}):Object.keys(h).forEach(N=>{const[w]=wt(N),k=i[w];k!=null&&(k.signatureKey=h[N],c.push(k))}),Object.keys(p).length>0?Object.keys(p).forEach(N=>{const[w]=wt(N),k=i[w];k&&(k.signatureKey=p[N],u.push(k))}):u=r;let y={};t.library!=null&&t.library.function!=null&&(y=t.library.function.reduce((N,w)=>(N[w.signature.name]=this.mapFunction(w),N),{}));const T={nodes:i,inputs:u,outputs:c,weights:a,placeholders:r,signature:n,functions:y};return o.length>0&&(T.initNodes=o),T}mapSignatureEntries(t){return Object.keys(t||{}).reduce((n,s)=>(n[t[s].name]=s,n),{})}mapNode(t){const n=Oa(t.op)||this.opMappers[t.op]||{};t.attr==null&&(t.attr={});const s={name:t.name,op:t.op,category:n.category,inputNames:(t.input||[]).map(r=>r.startsWith("^")?r.slice(1):r),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:t.attr,outputs:n.outputs};return n.inputs!=null&&(s.inputParams=n.inputs.reduce((r,a)=>(r[a.name]={type:a.type,inputIndexStart:a.start,inputIndexEnd:a.end},r),{})),n.attrs!=null&&(s.attrParams=n.attrs.reduce((r,a)=>{const o=a.type;let i;switch(a.type){case"string":i=qn(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=qn(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"string[]":i=Zn(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Zn(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"number":i=Gn(t.attr,a.tfName,a.defaultValue||0),i===void 0&&a.tfDeprecatedName&&(i=Gn(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"number[]":i=Jn(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Jn(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"bool":i=Hn(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Hn(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"bool[]":i=ts(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=ts(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"shape":i=Xn(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Xn(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"shape[]":i=Qn(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Qn(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"dtype":i=Mn(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Mn(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"dtype[]":i=Yn(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Yn(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"func":i=Zs(t.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Zs(t.attr,a.tfDeprecatedName,a.defaultValue));break;case"tensor":case"tensors":break;default:throw new Error(`Unsupported param type: ${a.type} for op: ${t.op}`)}return r[a.name]={value:i,type:o},r},{})),s}mapFunction(t){const n=t.nodeDef,s=[],r=[];let a={};n!=null&&(a=n.reduce((h,m)=>(h[m.name]=this.mapNode(m),m.op==="Const"&&r.push(h[m.name]),h),{}));const o=[],i=[];t.signature.inputArg.forEach(h=>{const[m]=wt(h.name),y={name:m,op:"Placeholder",inputs:[],inputNames:[],category:"graph",inputParams:{},attrParams:{dtype:{value:Os(h.type),type:"dtype"}},children:[]};y.signatureKey=h.name,o.push(y),a[m]=y}),Object.keys(a).forEach(h=>{const m=a[h];m.inputNames.forEach((y,T)=>{const[N,,w]=wt(y),k=a[N];if(k.outputs!=null){const v=k.outputs.indexOf(w);if(v!==-1){const $=`${N}:${v}`;m.inputNames[T]=$}}m.inputs.push(k),k.children.push(m)})});const c=t.ret;t.signature.outputArg.forEach(h=>{const[m,y]=wt(c[h.name]),T=a[m];T!=null&&(T.defaultOutput=y,i.push(T))});const p=this.mapArgsToSignature(t);return{nodes:a,inputs:o,outputs:i,weights:r,placeholders:s,signature:p}}mapArgsToSignature(t){return{methodName:t.signature.name,inputs:t.signature.inputArg.reduce((n,s)=>(n[s.name]=this.mapArgToTensorInfo(s),n),{}),outputs:t.signature.outputArg.reduce((n,s)=>(n[s.name]=this.mapArgToTensorInfo(s,t.ret),n),{})}}mapArgToTensorInfo(t,n){let s=t.name;return n!=null&&(s=n[s]),{name:s,dtype:t.type}}}function Xb(e){const t=x().global;if(typeof t.atob<"u")return t.atob(e);if(typeof Buffer<"u")return new Buffer(e,"base64").toString();throw new Error("Unable to decode base64 in this environment. Missing built-in atob() or Buffer()")}function Ca(e,t){const n=Array.isArray(e)?String.fromCharCode.apply(null,e):Xb(e);return t?n:n.toLowerCase()}function qn(e,t,n,s=!1){const r=e[t];return r!=null?Ca(r.s,s):n}function Hn(e,t,n){const s=e[t];return s?s.b:n}function Gn(e,t,n){const s=e[t]||{},r=s.i!=null?s.i:s.f!=null?s.f:n;return typeof r=="number"?r:parseInt(r,10)}function Os(e){switch(typeof e=="string"&&(e=pt[e]),e){case pt.DT_FLOAT:case pt.DT_HALF:return"float32";case pt.DT_INT32:case pt.DT_INT64:case pt.DT_INT8:case pt.DT_UINT8:return"int32";case pt.DT_BOOL:return"bool";case pt.DT_DOUBLE:return"float32";case pt.DT_STRING:return"string";default:return null}}function Zs(e,t,n){const s=e[t];return s&&s.func?s.func.name:n}function Mn(e,t,n){const s=e[t];return s&&s.type?Os(s.type):n}function Yn(e,t,n){const s=e[t];return s&&s.list&&s.list.type?s.list.type.map(r=>Os(r)):n}function Fa(e){if(!e.unknownRank)return e.dim!=null?e.dim.map(t=>typeof t.size=="number"?t.size:parseInt(t.size,10)):[]}function Xn(e,t,n){const s=e[t];return s&&s.shape?Fa(s.shape):n}function Jn(e,t,n){const s=e[t];return s?((s.list.f&&s.list.f.length?s.list.f:s.list.i)||[]).map(r=>typeof r=="number"?r:parseInt(r,10)):n}function Zn(e,t,n,s=!1){const r=e[t];return r&&r.list&&r.list.s?r.list.s.map(a=>Ca(a,s)):n}function Qn(e,t,n){const s=e[t];return s&&s.list&&s.list.shape?s.list.shape.map(r=>Fa(r)):n}function ts(e,t,n){const s=e[t];return s&&s.list&&s.list.b?s.list.b:n}/**
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
 */class Jb{constructor(t,n,s){this.node=t,this.tensorMap=n,this.context=s,this.inputs=[],this.attrs={},this.inputs=t.inputNames.map(r=>this.getInput(r)),t.rawAttrs!=null&&(this.attrs=Object.keys(t.rawAttrs).reduce((r,a)=>(r[a]=this.getAttr(a),r),{}))}getInput(t){return nt(t,this.tensorMap,this.context)}getAttr(t,n){const s=this.node.rawAttrs[t];if(s.tensor!=null)return nt(t,this.tensorMap,this.context);if(s.i!=null||s.f!=null)return Gn(this.node.rawAttrs,t,n);if(s.s!=null)return qn(this.node.rawAttrs,t,n);if(s.b!=null)return Hn(this.node.rawAttrs,t,n);if(s.shape!=null)return Xn(this.node.rawAttrs,t,n);if(s.type!=null)return Mn(this.node.rawAttrs,t,n);if(s.list!=null){if(s.list.i!=null||s.list.f!=null)return Jn(this.node.rawAttrs,t,n);if(s.list.s!=null)return Zn(this.node.rawAttrs,t,n);if(s.list.shape!=null)return Qn(this.node.rawAttrs,t,n);if(s.list.b!=null)return ts(this.node.rawAttrs,t,n);if(s.list.type!=null)return Yn(this.node.rawAttrs,t,n)}return n}}/**
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
 */const J=Object.freeze(Object.defineProperty({__proto__:null,OP_SCOPE_SUFFIX:xr,abs:ht,acos:Wl,acosh:Hl,add:Y,addN:Ml,all:Xl,any:Zl,argMax:tp,argMin:np,asin:rp,asinh:op,atan:up,atan2:lp,atanh:hp,avgPool:Zr,avgPool3d:kp,basicLSTMCell:xp,batchNorm:un,batchNorm2d:Fp,batchNorm3d:Pp,batchNorm4d:Rp,batchToSpaceND:Qr,bincount:ta,booleanMaskAsync:tg,broadcastArgs:jp,broadcastTo:Ke,buffer:_t,cast:H,ceil:Wp,clipByValue:Hp,clone:Ct,complex:Bt,concat:rt,concat1d:Mp,concat2d:Xp,concat3d:Zp,concat4d:th,conv1d:sh,conv2d:ln,conv2dTranspose:oh,conv3d:uh,conv3dTranspose:hh,cos:mh,cosh:gh,cosineWindow:Es,cumprod:bh,cumsum:Nh,denseBincount:Sh,depthToSpace:$h,depthwiseConv2d:gs,diag:vh,dilation2d:xh,div:st,divNoNan:Fh,dot:Ph,dropout:hg,einsum:Rh,elu:sa,enclosingPowerOfTwo:xa,equal:na,erf:jh,euclideanNorm:Jh,exp:Zt,expandDims:jt,expm1:ef,eye:oa,fft:Ss,fill:cn,floor:ia,floorDiv:Xr,fused:vg,gather:ua,gatherND:cg,greater:mn,greaterEqual:ca,ifft:Ze,imag:an,image:ab,inTopKAsync:mg,irfft:Ea,isFinite:lf,isInf:hf,isNaN:mf,leakyRelu:la,less:yf,lessEqual:ws,linalg:ob,linspace:wf,localResponseNormalization:Tf,log:xe,log1p:pa,logSigmoid:_f,logSoftmax:xf,logSumExp:fa,logicalAnd:Xe,logicalNot:ma,logicalOr:da,logicalXor:Bf,losses:ib,lowerBound:Lf,matMul:P,max:pe,maxPool:ga,maxPool3d:Vf,maxPoolWithArgmax:Kf,maximum:Wf,mean:Je,meshgrid:Hf,min:Un,minimum:ya,mirrorPad:Yf,mod:Jf,moments:Qf,movingAverage:ng,mul:D,multiRNNCell:em,multinomial:sm,neg:$t,norm:fn,notEqual:ba,oneHot:kl,ones:qt,onesLike:om,op:g,outerProduct:um,pad:Le,pad1d:pm,pad2d:fm,pad3d:dm,pad4d:ym,pool:Sm,pow:bs,prelu:Na,print:zr,prod:Em,raggedGather:vm,raggedTensorToTensor:xm,rand:Am,randomGamma:Bm,randomNormal:Ta,randomStandardNormal:Rm,randomUniform:Sa,range:De,real:Ie,reciprocal:jm,relu:gn,relu6:ka,reshape:S,reverse:Qt,reverse1d:Hm,reverse2d:Mm,reverse3d:Xm,reverse4d:Zm,rfft:ks,round:$a,rsqrt:ed,scalar:B,scatterND:rg,searchSorted:Ns,selu:sd,separableConv2d:ad,setdiff1dAsync:id,sigmoid:le,sign:cd,signal:rb,sin:pd,sinh:fd,slice:R,slice1d:dd,slice2d:yd,slice3d:wd,slice4d:Td,softmax:kd,softplus:ha,spaceToBatchND:wa,sparse:ub,sparseToDense:ig,spectral:sb,split:Ae,sqrt:Wn,square:hn,squaredDifference:_a,squeeze:$s,stack:It,step:va,stridedSlice:Fd,string:cb,sub:O,sum:j,tan:Pd,tanh:Kn,tensor:Nt,tensor1d:dt,tensor2d:$e,tensor3d:Hr,tensor4d:Ld,tensor5d:Rd,tensor6d:zd,tile:ke,topk:jd,transpose:Vn,truncatedNormal:Ud,unique:qd,unsortedSegmentSum:Gd,unstack:ne,upperBound:Yd,variable:Xd,where:de,whereAsync:Ia,zeros:ge,zerosLike:ys},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Zb=(e,t,n,s=J)=>{switch(e.op){case"BiasAdd":case"AddV2":case"Add":return[s.add(l("a",e,t,n),l("b",e,t,n))];case"AddN":return[s.addN(l("tensors",e,t,n))];case"FloorMod":case"Mod":return[s.mod(l("a",e,t,n),l("b",e,t,n))];case"Mul":return[s.mul(l("a",e,t,n),l("b",e,t,n))];case"RealDiv":case"Div":return[s.div(l("a",e,t,n),l("b",e,t,n))];case"DivNoNan":return[s.divNoNan(l("a",e,t,n),l("b",e,t,n))];case"FloorDiv":return[s.floorDiv(l("a",e,t,n),l("b",e,t,n))];case"Sub":return[s.sub(l("a",e,t,n),l("b",e,t,n))];case"Minimum":return[s.minimum(l("a",e,t,n),l("b",e,t,n))];case"Maximum":return[s.maximum(l("a",e,t,n),l("b",e,t,n))];case"Pow":return[s.pow(l("a",e,t,n),l("b",e,t,n))];case"SquaredDifference":return[s.squaredDifference(l("a",e,t,n),l("b",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const Qb=(e,t,n,s=J)=>{switch(e.op){case"Abs":case"ComplexAbs":return[s.abs(l("x",e,t,n))];case"Acos":return[s.acos(l("x",e,t,n))];case"Acosh":return[s.acosh(l("x",e,t,n))];case"Asin":return[s.asin(l("x",e,t,n))];case"Asinh":return[s.asinh(l("x",e,t,n))];case"Atan":return[s.atan(l("x",e,t,n))];case"Atan2":return[s.atan2(l("x",e,t,n),l("y",e,t,n))];case"Atanh":return[s.atanh(l("x",e,t,n))];case"Ceil":return[s.ceil(l("x",e,t,n))];case"Complex":return[s.complex(l("real",e,t,n),l("imag",e,t,n))];case"Cos":return[s.cos(l("x",e,t,n))];case"Cosh":return[s.cosh(l("x",e,t,n))];case"Elu":return[s.elu(l("x",e,t,n))];case"Erf":return[s.erf(l("x",e,t,n))];case"Exp":return[s.exp(l("x",e,t,n))];case"Expm1":return[s.expm1(l("x",e,t,n))];case"Floor":return[s.floor(l("x",e,t,n))];case"Log":return[s.log(l("x",e,t,n))];case"Log1p":return[s.log1p(l("x",e,t,n))];case"Imag":return[s.imag(l("x",e,t,n))];case"Neg":return[s.neg(l("x",e,t,n))];case"Reciprocal":return[s.reciprocal(l("x",e,t,n))];case"Real":return[s.real(l("x",e,t,n))];case"Relu":return[s.relu(l("x",e,t,n))];case"Round":return[s.round(l("x",e,t,n))];case"Selu":return[s.selu(l("x",e,t,n))];case"Sigmoid":return[s.sigmoid(l("x",e,t,n))];case"Sin":return[s.sin(l("x",e,t,n))];case"Sign":return[s.sign(l("x",e,t,n))];case"Sinh":return[s.sinh(l("x",e,t,n))];case"Softplus":return[s.softplus(l("x",e,t,n))];case"Sqrt":return[s.sqrt(l("x",e,t,n))];case"Square":return[s.square(l("x",e,t,n))];case"Tanh":return[s.tanh(l("x",e,t,n))];case"Tan":return[s.tan(l("x",e,t,n))];case"ClipByValue":return[s.clipByValue(l("x",e,t,n),l("clipValueMin",e,t,n),l("clipValueMax",e,t,n))];case"Relu6":return[s.relu6(l("x",e,t,n))];case"Rsqrt":return[s.rsqrt(nt(e.inputNames[0],t,n))];case"Prod":return[s.prod(l("x",e,t,n),l("axes",e,t,n))];case"LeakyRelu":return[s.leakyRelu(l("x",e,t,n),l("alpha",e,t,n))];case"Prelu":return[s.prelu(l("x",e,t,n),l("alpha",e,t,n))];case"IsNan":return[s.isNaN(nt(e.inputNames[0],t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */function ft(e,t,n=""){if(!(typeof e=="number"||typeof t=="number")){d(e.length===t.length,()=>n+` Shapes ${e} and ${t} must match`);for(let s=0;s<e.length;s++){const r=e[s],a=t[s];d(r<0||a<0||r===a,()=>n+` Shapes ${e} and ${t} must match`)}}}function Qs(e){return!(typeof e=="number"||e.some(t=>t<0))}function we(e,t,n){let s=es(e,n);const r=!Qs(s);if(r&&t.length===0)throw new Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${s}`);if(r&&t.forEach(a=>{s=es(a.shape,s)}),!Qs(s))throw new Error(`Non-fully-defined elementShape: ${s}`);return s}function es(e,t){if(typeof e=="number")return t;if(typeof t=="number")return e;if(e.length!==t.length)throw new Error(`Incompatible ranks during merge: ${e} vs. ${t}`);const n=[];for(let s=0;s<e.length;++s){const r=e[s],a=t[s];if(r>=0&&a>=0&&r!==a)throw new Error(`Incompatible shape during merge: ${e} vs. ${t}`);n[s]=r>=0?r:a}return n}/**
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
 */class tw{constructor(t,n,s,r,a,o,i){this.name=t,this.dtype=n,this.maxSize=s,this.elementShape=r,this.identicalElementShapes=a,this.dynamicSize=o,this.clearAfterRead=i,this.tensors=[],this.closed_=!1,this.idTensor=B(0),At(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(t){this.tensors.forEach(n=>{(t==null||!t.has(n.tensor.id))&&n.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(t){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(t<0||t>=this.size())throw new Error(`Tried to read from index ${t}, but array size is: ${this.size()}`);const n=this.tensors[t];if(n.cleared)throw new Error(`TensorArray ${this.name}: Could not read index ${t} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(n.cleared=!0),n.read=!0,n.tensor}readMany(t){return t.map(n=>this.read(n))}write(t,n){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(t<0||!this.dynamicSize&&t>=this.maxSize)throw new Error(`Tried to write to index ${t}, but array is not resizeable and size is: ${this.maxSize}`);const s=this.tensors[t]||{};if(n.dtype!==this.dtype)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${t},
          because the value dtype is ${n.dtype}, but TensorArray dtype is ${this.dtype}.`);if(this.size()===0&&(this.elementShape==null||this.elementShape.length===0)&&(this.elementShape=n.shape),ft(this.elementShape,n.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${t}.`),s.read)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${t}, because it has already been read.`);if(s.written)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${t}, because it has already been written.`);s.tensor=n,At(n),s.written=!0,this.tensors[t]=s}writeMany(t,n){if(t.length!==n.length)throw new Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${t.length} is not the same as tensors size: ${n.length}.`);t.forEach((s,r)=>this.write(s,n[r]))}gather(t,n){if(n&&n!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${n}`);if(t)t=t.slice(0,this.size());else{t=[];for(let r=0;r<this.size();r++)t.push(r)}if(t.length===0)return Nt([],[0].concat(this.elementShape));const s=this.readMany(t);return ft(this.elementShape,s[0].shape,"TensorArray shape mismatch: "),It(s,0)}concat(t){if(t&&t!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${t}`);if(this.size()===0)return Nt([],[0].concat(this.elementShape));const n=[];for(let r=0;r<this.size();r++)n.push(r);const s=this.readMany(n);return ft(this.elementShape,s[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${s[0].shape})`),rt(s,0)}scatter(t,n){if(n.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${n.dtype}`);if(t.length!==n.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${t.length} vs. ${n.shape[0]}`);const s=Math.max(...t);if(!this.dynamicSize&&s>=this.maxSize)throw new Error(`Max index must be < array size (${s}  vs. ${this.maxSize})`);this.writeMany(t,ne(n,0))}split(t,n){if(n.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${n.dtype}`);let s=0;const r=t.map(u=>(s+=u,s));if(s!==n.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${s}, and tensor's shape is: ${n.shape}`);if(!this.dynamicSize&&t.length!==this.maxSize)throw new Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${t.length}), and the TensorArray is not marked as dynamically resizeable`);const a=s===0?0:n.size/s,o=[];gt(()=>{n=S(n,[1,s,a]);for(let u=0;u<t.length;++u){const p=[0,u===0?0:r[u-1],0],h=[1,t[u],a];o[u]=S(R(n,p,h),this.elementShape)}return o});const i=[];for(let u=0;u<t.length;u++)i[u]=u;this.writeMany(i,o)}}/**
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
 */class te{constructor(t,n,s,r=-1){this.tensors=t,this.elementShape=n,this.elementDtype=s,t!=null&&t.forEach(a=>{if(s!==a.dtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${a.dtype}`);ft(n,a.shape,"TensorList shape mismatch: "),At(a)}),this.idTensor=B(0),this.maxNumElements=r,At(this.idTensor)}get id(){return this.idTensor.id}copy(){return new te([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(t){this.tensors.forEach(n=>{(t==null||!t.has(n.id))&&n.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(t,n,s=-1){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(s!==-1&&this.tensors.length!==s)throw new Error(`Operation expected a list with ${s} elements but got a list with ${this.tensors.length} elements.`);ft(t,this.elementShape,"TensorList shape mismatch: ");const r=we(this.elementShape,this.tensors,t);return gt(()=>{const a=this.tensors.map(o=>S(o,r));return It(a,0)})}popBack(t,n){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(this.size()===0)throw new Error("Trying to pop from an empty list.");const s=we(this.elementShape,this.tensors,t),r=this.tensors.pop();return r.kept=!1,ft(r.shape,t,"TensorList shape mismatch: "),S(r,s)}pushBack(t){if(t.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t.dtype}, but list elements ${this.elementDtype}`);if(ft(t.shape,this.elementShape,"TensorList shape mismatch: "),this.maxNumElements===this.size())throw new Error("Trying to push element into a full list.");At(t),this.tensors.push(t)}resize(t){if(t<0)throw new Error(`TensorListResize expects size to be non-negative. Got: ${t}`);if(this.maxNumElements!==-1&&t>this.maxNumElements)throw new Error(`TensorListResize input size ${t} is greater maxNumElement ${this.maxNumElements}.`);const n=new te([],this.elementShape,this.elementDtype,this.maxNumElements);n.tensors.length=t;for(let s=0;s<Math.min(this.tensors.length,t);++s)n.tensors[s]=this.tensors[s];return n}getItem(t,n,s){if(s!==this.elementDtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${this.elementDtype}`);if(t<0||t>this.tensors.length)throw new Error(`Trying to access element ${t} in a list with ${this.tensors.length} elements.`);if(this.tensors[t]==null)throw new Error(`element at index ${t} is null.`);ft(this.tensors[t].shape,n,"TensorList shape mismatch: ");const r=we(this.elementShape,this.tensors,n);return S(this.tensors[t],r)}setItem(t,n){if(n.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n.dtype}, but list elements ${this.elementDtype}`);if(t<0||this.maxNumElements!==-1&&t>=this.maxNumElements)throw new Error(`Trying to set element ${t} in a list with max ${this.maxNumElements} elements.`);ft(this.elementShape,n.shape,"TensorList shape mismatch: "),At(n),this.tensors[t]!=null&&(this.tensors[t].kept=!1),this.tensors[t]=n}gather(t,n,s){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);ft(this.elementShape,s,"TensorList shape mismatch: "),t=t.slice(0,this.size());const r=we(this.elementShape,this.tensors,s);return t.length===0?Nt([],[0].concat(r)):gt(()=>{const a=t.map(o=>S(this.tensors[o],r));return It(a,0)})}concat(t,n){if(t&&t!==this.elementDtype)throw new Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${t}`);ft(this.elementShape,n,"TensorList shape mismatch: ");const s=we(this.elementShape,this.tensors,n);return this.size()===0?Nt([],[0].concat(s)):gt(()=>{const r=this.tensors.map(a=>S(a,s));return rt(r,0)})}}function ew(e,t,n){const s=e.dtype;if(e.shape.length<1)throw new Error(`Tensor must be at least a vector, but saw shape: ${e.shape}`);if(e.dtype!==n)throw new Error(`Invalid data types; op elements ${e.dtype}, but list elements ${n}`);const r=e.shape.slice(1);ft(r,t,"TensorList shape mismatch: ");const a=ne(e);return new te(a,t,s)}function nw(e,t,n,s){return new te([],e,t,s)}function sw(e,t,n,s){if(t.length!==e.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${t.length} vs. ${e.shape[0]}`);const r=Math.max(...t);if(s!=null&&s!==-1&&r>=s)throw new Error(`Max index must be < array size (${r}  vs. ${s})`);const a=new te([],n,e.dtype,s),o=ne(e,0);return t.forEach((i,u)=>{a.setItem(i,o[u])}),a}function rw(e,t,n){let s=0;const r=t.map(p=>(s+=p,s));if(s!==e.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${s}, and tensor's shape is: ${e.shape}`);const a=e.shape.slice(1),o=es(a,n),i=s===0?0:e.size/s,u=gt(()=>{const p=[];e=S(e,[1,s,i]);for(let h=0;h<t.length;++h){const y=[0,h===0?0:r[h-1],0],T=[1,t[h],i];p[h]=S(R(e,y,T),o)}return e.dispose(),p}),c=new te([],n,e.dtype,t.length);for(let p=0;p<u.length;p++)c.setItem(p,u[p]);return c}/**
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
 */const aw=async(e,t,n)=>{switch(e.op){case"If":case"StatelessIf":{const s=l("thenBranch",e,t,n),r=l("elseBranch",e,t,n),a=l("cond",e,t,n),o=l("args",e,t,n);return(await a.data())[0]?n.functionMap[s].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap):n.functionMap[r].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap)}case"While":case"StatelessWhile":{const s=l("body",e,t,n),r=l("cond",e,t,n),a=l("args",e,t,n),o=await n.functionMap[r].executeFunctionAsync(a,n.tensorArrayMap,n.tensorListMap),i=a.map(p=>p.id);let u=await o[0].data();o.forEach(p=>{!p.kept&&i.indexOf(p.id)===-1&&p.dispose()});let c=a;for(;u[0];){const p=c;c=await n.functionMap[s].executeFunctionAsync(c,n.tensorArrayMap,n.tensorListMap);const h=c.map(y=>y.id);p.forEach(y=>{!y.kept&&i.indexOf(y.id)===-1&&h.indexOf(y.id)===-1&&y.dispose()});const m=await n.functionMap[r].executeFunctionAsync(c,n.tensorArrayMap,n.tensorListMap);u=await m[0].data(),m.forEach(y=>{!y.kept&&i.indexOf(y.id)===-1&&h.indexOf(y.id)===-1&&y.dispose()})}return c}case"LoopCond":{const s=l("pred",e,t,n);return[St(s)]}case"Switch":{const s=l("pred",e,t,n);let r=l("data",e,t,n);return r.kept||(r=St(r)),(await s.data())[0]?[void 0,r]:[r,void 0]}case"Merge":{const s=e.inputNames.find(r=>nt(r,t,n)!==void 0);if(s){const r=nt(s,t,n);return[St(r)]}return}case"Enter":{const s=l("frameName",e,t,n),r=l("tensor",e,t,n);return n.enterFrame(s),[St(r)]}case"Exit":{const s=l("tensor",e,t,n);return n.exitFrame(),[St(s)]}case"NextIteration":{const s=l("tensor",e,t,n);return n.nextIteration(),[St(s)]}case"TensorArrayV3":{const s=l("size",e,t,n),r=l("dtype",e,t,n),a=l("elementShape",e,t,n),o=l("dynamicSize",e,t,n),i=l("clearAfterRead",e,t,n),u=l("identicalElementShapes",e,t,n),c=l("name",e,t,n),p=new tw(c,r,s,a,u,o,i);return n.addTensorArray(p),[p.idTensor,B(1)]}case"TensorArrayWriteV3":{const s=l("tensorArrayId",e,t,n),r=l("index",e,t,n),a=l("tensor",e,t,n),o=n.getTensorArray(s.id);return o.write(r,a),[o.idTensor]}case"TensorArrayReadV3":{const s=l("tensorArrayId",e,t,n),r=l("index",e,t,n);return[n.getTensorArray(s.id).read(r)]}case"TensorArrayGatherV3":{const s=l("tensorArrayId",e,t,n),r=l("indices",e,t,n),a=l("dtype",e,t,n);return[n.getTensorArray(s.id).gather(r,a)]}case"TensorArrayScatterV3":{const s=l("tensorArrayId",e,t,n),r=l("indices",e,t,n),a=l("tensor",e,t,n),o=n.getTensorArray(s.id);return o.scatter(r,a),[o.idTensor]}case"TensorArrayConcatV3":{const s=l("tensorArrayId",e,t,n),r=n.getTensorArray(s.id),a=l("dtype",e,t,n);return[r.concat(a)]}case"TensorArraySplitV3":{const s=l("tensorArrayId",e,t,n),r=l("tensor",e,t,n),a=l("lengths",e,t,n),o=n.getTensorArray(s.id);return o.split(a,r),[o.idTensor]}case"TensorArraySizeV3":{const s=l("tensorArrayId",e,t,n),r=n.getTensorArray(s.id);return[B(r.size(),"int32")]}case"TensorArrayCloseV3":{const s=l("tensorArrayId",e,t,n),r=n.getTensorArray(s.id);return r.clearAndClose(),[r.idTensor]}case"TensorListSetItem":{const s=l("tensorListId",e,t,n),r=l("index",e,t,n),a=l("tensor",e,t,n),o=n.getTensorList(s.id);return o.setItem(r,a),[o.idTensor]}case"TensorListGetItem":{const s=l("tensorListId",e,t,n),r=l("index",e,t,n),a=l("elementShape",e,t,n),o=l("elementDType",e,t,n);return[n.getTensorList(s.id).getItem(r,a,o)]}case"TensorListScatterV2":case"TensorListScatter":{const s=l("indices",e,t,n),r=l("tensor",e,t,n),a=l("elementShape",e,t,n),o=l("numElements",e,t,n),i=sw(r,s,a,o);return n.addTensorList(i),[i.idTensor]}case"TensorListReserve":case"EmptyTensorList":{const s=l("elementShape",e,t,n),r=l("elementDType",e,t,n);let a;e.op==="TensorListReserve"?a="numElements":a="maxNumElements";const o=l(a,e,t,n),i=e.op==="TensorListReserve"?-1:o,u=nw(s,r,o,i);return n.addTensorList(u),[u.idTensor]}case"TensorListGather":{const s=l("tensorListId",e,t,n),r=l("indices",e,t,n),a=l("elementShape",e,t,n),o=l("elementDType",e,t,n);return[n.getTensorList(s.id).gather(r,o,a)]}case"TensorListStack":{const s=l("tensorListId",e,t,n),r=l("elementShape",e,t,n),a=l("elementDType",e,t,n),o=l("numElements",e,t,n);return[n.getTensorList(s.id).stack(r,a,o)]}case"TensorListFromTensor":{const s=l("tensor",e,t,n),r=l("elementShape",e,t,n),a=l("elementDType",e,t,n),o=ew(s,r,a);return n.addTensorList(o),[o.idTensor]}case"TensorListConcat":case"TensorListConcatV2":{const s=l("tensorListId",e,t,n),r=n.getTensorList(s.id),a=l("dtype",e,t,n),o=l("elementShape",e,t,n);return[r.concat(a,o)]}case"TensorListPushBack":{const s=l("tensorListId",e,t,n),r=l("tensor",e,t,n),a=n.getTensorList(s.id);return a.pushBack(r),[a.idTensor]}case"TensorListPopBack":{const s=l("tensorListId",e,t,n),r=l("elementShape",e,t,n),a=l("elementDType",e,t,n);return[n.getTensorList(s.id).popBack(r,a)]}case"TensorListSplit":{const s=l("tensor",e,t,n),r=l("elementShape",e,t,n),a=l("lengths",e,t,n),o=rw(s,a,r);return n.addTensorList(o),[o.idTensor]}case"TensorListLength":{const s=l("tensorListId",e,t,n),r=n.getTensorList(s.id);return[B(r.size(),"int32")]}case"TensorListResize":{const s=l("tensorListId",e,t,n),r=l("size",e,t,n),o=n.getTensorList(s.id).resize(r);return n.addTensorList(o),[o.idTensor]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */function tr(e,t,n){const[s,r]=l("fusedOps",e,t,n),a=s==="biasadd",o=!a,i=r==="prelu",u=s==="fusedbatchnorm",c=l("numArgs",e,t,n);if(a){if(i&&c!==2)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!i&&a&&c!==1)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.")}if(u)throw new Error("FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported");const p=l("strides",e,t,n),h=Ue(e,t,n),m=l("dataFormat",e,t,n).toUpperCase(),y=l("dilations",e,t,n);let[T,N]=l("args",e,t,n);o&&(N=T,T=void 0);const w=l("leakyreluAlpha",e,t,n);return{stride:p,pad:h,dataFormat:m,dilations:y,biasArg:T,preluArg:N,activationFunc:r,leakyreluAlpha:w}}const ow=(e,t,n,s=J)=>{switch(e.op){case"Conv1D":{const r=l("stride",e,t,n),a=l("pad",e,t,n),o=l("dataFormat",e,t,n).toUpperCase(),i=l("dilation",e,t,n);return[s.conv1d(l("x",e,t,n),l("filter",e,t,n),r,a,o,i)]}case"Conv2D":{const r=l("strides",e,t,n),a=Ue(e,t,n),o=l("dataFormat",e,t,n).toUpperCase(),i=l("dilations",e,t,n);return[s.conv2d(l("x",e,t,n),l("filter",e,t,n),[r[1],r[2]],a,o,[i[1],i[2]])]}case"_FusedConv2D":{const{stride:r,pad:a,dataFormat:o,dilations:i,biasArg:u,preluArg:c,activationFunc:p,leakyreluAlpha:h}=tr(e,t,n);return[s.fused.conv2d({x:l("x",e,t,n),filter:l("filter",e,t,n),strides:[r[1],r[2]],pad:a,dataFormat:o,dilations:[i[1],i[2]],bias:u,activation:p,preluActivationWeights:c,leakyreluAlpha:h})]}case"FusedDepthwiseConv2dNative":{const{stride:r,pad:a,dataFormat:o,dilations:i,biasArg:u,preluArg:c,activationFunc:p,leakyreluAlpha:h}=tr(e,t,n);return[s.fused.depthwiseConv2d({x:l("x",e,t,n),filter:l("filter",e,t,n),strides:[r[1],r[2]],pad:a,dataFormat:o,dilations:[i[1],i[2]],bias:u,activation:p,preluActivationWeights:c,leakyreluAlpha:h})]}case"Conv2DBackpropInput":case"Conv2dTranspose":{const r=l("outputShape",e,t,n),a=l("strides",e,t,n),o=Ue(e,t,n);return[s.conv2dTranspose(l("x",e,t,n),l("filter",e,t,n),r,[a[1],a[2]],o)]}case"DepthwiseConv2dNative":case"DepthwiseConv2d":{const r=l("strides",e,t,n),a=Ue(e,t,n),o=l("dilations",e,t,n),i=l("dataFormat",e,t,n).toUpperCase();return[s.depthwiseConv2d(l("input",e,t,n),l("filter",e,t,n),[r[1],r[2]],a,i,[o[1],o[2]])]}case"Conv3D":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("dataFormat",e,t,n).toUpperCase(),i=l("dilations",e,t,n);return[s.conv3d(l("x",e,t,n),l("filter",e,t,n),[r[1],r[2],r[3]],a,o,[i[1],i[2],i[3]])]}case"AvgPool":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("kernelSize",e,t,n);return[s.avgPool(l("x",e,t,n),[o[1],o[2]],[r[1],r[2]],a)]}case"MaxPool":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("kernelSize",e,t,n);return[s.maxPool(l("x",e,t,n),[o[1],o[2]],[r[1],r[2]],a)]}case"MaxPoolWithArgmax":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("kernelSize",e,t,n),i=l("includeBatchInIndex",e,t,n),{result:u,indexes:c}=s.maxPoolWithArgmax(l("x",e,t,n),[o[1],o[2]],[r[1],r[2]],a,i);return[u,c]}case"AvgPool3D":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("kernelSize",e,t,n);return[s.avgPool3d(l("x",e,t,n),[o[1],o[2],o[3]],[r[1],r[2],r[3]],a)]}case"MaxPool3D":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("kernelSize",e,t,n);return[s.maxPool3d(l("x",e,t,n),[o[1],o[2],o[3]],[r[1],r[2],r[3]],a)]}case"Dilation2D":{const r=l("strides",e,t,n),a=l("pad",e,t,n),o=l("dilations",e,t,n),i=r[1],u=r[2],c=o[1],p=o[2];return[s.dilation2d(l("x",e,t,n),l("filter",e,t,n),[i,u],a,[c,p],"NHWC")]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const iw=(e,t,n,s=J)=>{switch(e.op){case"Fill":{const r=l("shape",e,t,n),a=l("dtype",e,t,n),o=l("value",e,t,n);return[s.fill(r,o,a)]}case"LinSpace":{const r=l("start",e,t,n),a=l("stop",e,t,n),o=l("num",e,t,n);return[s.linspace(r,a,o)]}case"Multinomial":{const r=l("logits",e,t,n),a=l("numSamples",e,t,n),o=l("seed",e,t,n);return[s.multinomial(r,a,o)]}case"OneHot":{const r=l("indices",e,t,n),a=l("depth",e,t,n),o=l("onValue",e,t,n),i=l("offValue",e,t,n),u=l("dtype",e,t,n);return[s.oneHot(r,a,o,i,u)]}case"Ones":return[s.ones(l("shape",e,t,n),l("dtype",e,t,n))];case"OnesLike":return[s.onesLike(l("x",e,t,n))];case"RandomStandardNormal":return[s.randomStandardNormal(l("shape",e,t,n),l("dtype",e,t,n),l("seed",e,t,n))];case"RandomUniform":return[s.randomUniform(l("shape",e,t,n),l("minval",e,t,n),l("maxval",e,t,n),l("dtype",e,t,n))];case"Range":{const r=l("start",e,t,n),a=l("stop",e,t,n),o=l("step",e,t,n);return[s.range(r,a,o,l("dtype",e,t,n))]}case"TruncatedNormal":{const r=l("shape",e,t,n),a=l("mean",e,t,n),o=l("stdDev",e,t,n),i=l("seed",e,t,n);return[s.truncatedNormal(r,a,o,l("dtype",e,t,n),i)]}case"Zeros":return[s.zeros(l("shape",e,t,n),l("dtype",e,t,n))];case"ZerosLike":return[s.zerosLike(l("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */function kn(e,t,n){const s=l("boxes",e,t,n),r=l("scores",e,t,n),a=l("maxOutputSize",e,t,n),o=l("iouThreshold",e,t,n),i=l("scoreThreshold",e,t,n),u=l("softNmsSigma",e,t,n);return{boxes:s,scores:r,maxOutputSize:a,iouThreshold:o,scoreThreshold:i,softNmsSigma:u}}const uw=async(e,t,n,s,r=J)=>{switch(e.op){case"NonMaxSuppressionV5":{const{boxes:a,scores:o,maxOutputSize:i,iouThreshold:u,scoreThreshold:c,softNmsSigma:p}=kn(e,t,n),h=await r.image.nonMaxSuppressionWithScoreAsync(a,o,i,u,c,p);return[h.selectedIndices,h.selectedScores]}case"NonMaxSuppressionV4":{const{boxes:a,scores:o,maxOutputSize:i,iouThreshold:u,scoreThreshold:c}=kn(e,t,n),p=l("padToMaxOutputSize",e,t,n),h=await r.image.nonMaxSuppressionPaddedAsync(a,o,i,u,c,p);return[h.selectedIndices,h.validOutputs]}case"NonMaxSuppressionV3":case"NonMaxSuppressionV2":{const{boxes:a,scores:o,maxOutputSize:i,iouThreshold:u,scoreThreshold:c}=kn(e,t,n);return[await r.image.nonMaxSuppressionAsync(a,o,i,u,c)]}case"Where":{const a=r.cast(l("condition",e,t,n),"bool"),o=[await r.whereAsync(a)];return a.dispose(),o}case"ListDiff":return r.setdiff1dAsync(l("x",e,t,n),l("y",e,t,n));default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const cw=(e,t,n,s=J)=>{switch(e.op){case"LowerBound":{const r=l("sortedSequence",e,t,n),a=l("values",e,t,n);return[s.lowerBound(r,a)]}case"TopKV2":{const r=l("x",e,t,n),a=l("k",e,t,n),o=l("sorted",e,t,n),i=s.topk(r,a,o);return[i.values,i.indices]}case"UpperBound":{const r=l("sortedSequence",e,t,n),a=l("values",e,t,n);return[s.upperBound(r,a)]}case"Unique":{const r=l("x",e,t,n),a=s.unique(r);return[a.values,a.indices]}case"UniqueV2":{const r=l("x",e,t,n),a=l("axis",e,t,n),o=s.unique(r,a);return[o.values,o.indices]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const lw=(e,t,n,s=J)=>{switch(e.op){case"Const":return t[e.name];case"PlaceholderWithDefault":const r=l("default",e,t,n);return[nt(e.name,t,n)||r];case"Placeholder":return[nt(e.name,t,n)];case"Identity":case"StopGradient":case"FakeQuantWithMinMaxVars":{const p=l("x",e,t,n);return[St(p)]}case"IdentityN":return l("x",e,t,n).map(p=>St(p));case"Snapshot":const a=l("x",e,t,n);return[St(a)];case"Shape":return[s.tensor1d(l("x",e,t,n).shape,"int32")];case"ShapeN":return l("x",e,t,n).map(p=>s.tensor1d(p.shape));case"Size":return[s.scalar(l("x",e,t,n).size,"int32")];case"Rank":return[s.scalar(l("x",e,t,n).rank,"int32")];case"NoOp":return[s.scalar(1)];case"Print":const o=l("x",e,t,n),i=l("data",e,t,n),u=l("message",e,t,n),c=l("summarize",e,t,n);console.warn("The graph has a tf.print() operation,usually used for debugging, which slows down performance."),console.log(u);for(let p=0;p<i.length;p++)console.log(Array.prototype.slice.call(i[p].dataSync()).slice(0,c));return[o];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */class pw{constructor(t,n){this.keyDType=t,this.valueDType=n,this.handle=B(0),this.tensorMap=new Map,At(this.handle)}get id(){return this.handle.id}clearAndClose(){this.tensorMap.forEach(t=>t.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return B(this.size(),"int32")}async import(t,n){this.checkKeyAndValueTensor(t,n);const s=await t.data();return this.tensorMap.forEach(r=>r.dispose()),this.tensorMap.clear(),gt(()=>{const r=ne(n),a=s.length,o=r.length;d(a===o,()=>`The number of elements doesn't match, keys has ${a} elements, the values has ${o} elements.`);for(let i=0;i<a;i++){const u=s[i],c=r[i];At(c),this.tensorMap.set(u,c)}return this.handle})}async find(t,n){this.checkKeyAndValueTensor(t,n);const s=await t.data();return gt(()=>{const r=[];for(let a=0;a<s.length;a++){const o=s[a],i=this.findWithDefault(o,n);r.push(i)}return It(r)})}findWithDefault(t,n){const s=this.tensorMap.get(t);return s??n}checkKeyAndValueTensor(t,n){if(t.dtype!==this.keyDType)throw new Error(`Expect key dtype ${this.keyDType}, but got ${t.dtype}`);if(n.dtype!==this.valueDType)throw new Error(`Expect value dtype ${this.valueDType}, but got ${n.dtype}`)}}/**
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
 */const hw=async(e,t,n,s)=>{switch(e.op){case"HashTable":case"HashTableV2":{const r=l("keyDType",e,t,n),a=l("valueDType",e,t,n),o=new pw(r,a);return s.addHashTable(e.name,o),[o.handle]}case"LookupTableImport":case"LookupTableImportV2":{const r=l("tableHandle",e,t,n,s),a=l("keys",e,t,n),o=l("values",e,t,n);return[await s.getHashTableById(r.id).import(a,o)]}case"LookupTableFind":case"LookupTableFindV2":{const r=l("tableHandle",e,t,n,s),a=l("keys",e,t,n),o=l("defaultValue",e,t,n);return[await s.getHashTableById(r.id).find(a,o)]}case"LookupTableSize":case"LookupTableSizeV2":{const r=l("tableHandle",e,t,n,s);return[s.getHashTableById(r.id).tensorSize()]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const fw=(e,t,n,s=J)=>{switch(e.op){case"ResizeBilinear":{const r=l("images",e,t,n),a=l("size",e,t,n),o=l("alignCorners",e,t,n),i=l("halfPixelCenters",e,t,n);return[s.image.resizeBilinear(r,[a[0],a[1]],o,i)]}case"ResizeNearestNeighbor":{const r=l("images",e,t,n),a=l("size",e,t,n),o=l("alignCorners",e,t,n),i=l("halfPixelCenters",e,t,n);return[s.image.resizeNearestNeighbor(r,[a[0],a[1]],o,i)]}case"CropAndResize":{const r=l("image",e,t,n),a=l("boxes",e,t,n),o=l("boxInd",e,t,n),i=l("cropSize",e,t,n),u=l("method",e,t,n),c=l("extrapolationValue",e,t,n);return[s.image.cropAndResize(r,a,o,i,u,c)]}case"ImageProjectiveTransformV3":{const r=l("images",e,t,n),a=l("transforms",e,t,n),o=l("outputShape",e,t,n),i=l("fillValue",e,t,n),u=l("interpolation",e,t,n),c=l("fillMode",e,t,n);return[s.image.transform(r,a,u.toLowerCase(),c.toLowerCase(),i,o)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const mw=(e,t,n,s=J)=>{switch(e.op){case"Equal":return[s.equal(l("a",e,t,n),l("b",e,t,n))];case"NotEqual":return[s.notEqual(l("a",e,t,n),l("b",e,t,n))];case"Greater":return[s.greater(l("a",e,t,n),l("b",e,t,n))];case"GreaterEqual":return[s.greaterEqual(l("a",e,t,n),l("b",e,t,n))];case"Less":return[s.less(l("a",e,t,n),l("b",e,t,n))];case"LessEqual":return[s.lessEqual(l("a",e,t,n),l("b",e,t,n))];case"LogicalAnd":return[s.logicalAnd(l("a",e,t,n),l("b",e,t,n))];case"LogicalNot":return[s.logicalNot(l("a",e,t,n))];case"LogicalOr":return[s.logicalOr(l("a",e,t,n),l("b",e,t,n))];case"Select":case"SelectV2":return[s.where(l("condition",e,t,n),l("a",e,t,n),l("b",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const dw=(e,t,n,s=J)=>{switch(e.op){case"BatchMatMul":case"BatchMatMulV2":case"MatMul":return[s.matMul(l("a",e,t,n),l("b",e,t,n),l("transposeA",e,t,n),l("transposeB",e,t,n))];case"Einsum":return[s.einsum(l("equation",e,t,n),...l("tensors",e,t,n))];case"Transpose":return[s.transpose(l("x",e,t,n),l("perm",e,t,n))];case"_FusedMatMul":const[r,a]=l("fusedOps",e,t,n),o=r==="biasadd",i=a==="prelu",u=l("numArgs",e,t,n),c=l("leakyreluAlpha",e,t,n);if(o){if(i&&u!==2)throw new Error("Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!i&&u!==1)throw new Error("Fused MatMul with BiasAdd must have one extra argument: bias.")}const[p,h]=l("args",e,t,n);return[s.fused.matMul({a:l("a",e,t,n),b:l("b",e,t,n),transposeA:l("transposeA",e,t,n),transposeB:l("transposeB",e,t,n),bias:p,activation:a,preluActivationWeights:h,leakyreluAlpha:c})];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const gw=(e,t,n,s=J)=>{switch(e.op){case"EuclideanNorm":return[s.euclideanNorm(l("x",e,t,n),l("axis",e,t,n),l("keepDims",e,t,n))];case"FusedBatchNorm":case"FusedBatchNormV2":return[s.batchNorm(l("x",e,t,n),l("mean",e,t,n),l("variance",e,t,n),l("offset",e,t,n),l("scale",e,t,n),l("epsilon",e,t,n))];case"FusedBatchNormV3":return[s.batchNorm(l("x",e,t,n),l("mean",e,t,n),l("variance",e,t,n),l("offset",e,t,n),l("scale",e,t,n),l("epsilon",e,t,n))];case"LRN":return[s.localResponseNormalization(l("x",e,t,n),l("radius",e,t,n),l("bias",e,t,n),l("alpha",e,t,n),l("beta",e,t,n))];case"Softmax":return[s.softmax(l("x",e,t,n))];case"LogSoftmax":return[s.logSoftmax(l("x",e,t,n))];case"SparseToDense":return[s.sparseToDense(l("sparseIndices",e,t,n),l("outputShape",e,t,n),l("sparseValues",e,t,n),l("defaultValue",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const yw=(e,t,n,s=J)=>{switch(e.op){case"Max":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.max(l("x",e,t,n),i,u)]}case"Mean":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.mean(l("x",e,t,n),i,u)]}case"Min":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.min(l("x",e,t,n),i,u)]}case"Sum":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.sum(l("x",e,t,n),i,u)]}case"All":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.all(l("x",e,t,n),i,u)]}case"Any":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.any(l("x",e,t,n),i,u)]}case"ArgMax":{const i=l("axis",e,t,n);return[s.argMax(l("x",e,t,n),i)]}case"ArgMin":{const i=l("axis",e,t,n);return[s.argMin(l("x",e,t,n),i)]}case"Prod":{const i=l("axis",e,t,n),u=l("keepDims",e,t,n);return[s.prod(l("x",e,t,n),i,u)]}case"Cumprod":{const i=l("axis",e,t,n),u=l("exclusive",e,t,n),c=l("reverse",e,t,n);return[s.cumprod(l("x",e,t,n),i,u,c)]}case"Cumsum":{const i=l("axis",e,t,n),u=l("exclusive",e,t,n),c=l("reverse",e,t,n);return[s.cumsum(l("x",e,t,n),i,u,c)]}case"Bincount":const r=l("x",e,t,n),a=l("weights",e,t,n),o=l("size",e,t,n);return[s.bincount(r,a,o)];case"DenseBincount":{const i=l("x",e,t,n),u=l("weights",e,t,n),c=l("size",e,t,n),p=l("binaryOutput",e,t,n);return[s.denseBincount(i,u,c,p)]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const bw=(e,t,n,s=J)=>{switch(e.op){case"ConcatV2":case"Concat":{const r=l("n",e,t,n),a=l("axis",e,t,n);let o=l("tensors",e,t,n);return o=o.slice(0,r),[s.concat(o,a)]}case"Gather":{const r=l("x",e,t,n),a=l("indices",e,t,n);return[s.gather(r,s.cast(a,"int32"),0)]}case"GatherV2":{const r=l("axis",e,t,n),a=l("batchDims",e,t,n),o=l("x",e,t,n),i=l("indices",e,t,n);return[s.gather(o,s.cast(i,"int32"),r,a)]}case"Reverse":{const r=l("dims",e,t,n),a=[];for(let i=0;i<r.length;i++)r[i]&&a.push(i);const o=l("x",e,t,n);return[s.reverse(o,a)]}case"ReverseV2":{const r=l("axis",e,t,n),a=l("x",e,t,n);return[s.reverse(a,r)]}case"Slice":{const r=l("begin",e,t,n),a=l("size",e,t,n);return[s.slice(l("x",e,t,n),r,a)]}case"StridedSlice":{const r=l("begin",e,t,n),a=l("end",e,t,n),o=l("strides",e,t,n),i=l("beginMask",e,t,n),u=l("endMask",e,t,n),c=l("ellipsisMask",e,t,n),p=l("newAxisMask",e,t,n),h=l("shrinkAxisMask",e,t,n),m=l("x",e,t,n);return[s.stridedSlice(m,r,a,o,i,u,c,p,h)]}case"Pack":return gt(()=>{const r=l("axis",e,t,n),a=l("tensors",e,t,n),o=a[0].shape,i=s.squeeze(a[0]).shape,u=a.map(c=>{const p=Et(c.shape,o);if(!p&&!Et(s.squeeze(c).shape,i))throw new Error("the input tensors shape does not match");return p?c:s.reshape(c,o)});return[s.stack(u,r)]});case"Unpack":{const r=l("axis",e,t,n),a=l("tensor",e,t,n);return s.unstack(a,r)}case"Tile":{const r=l("reps",e,t,n);return[s.tile(l("x",e,t,n),r)]}case"Split":case"SplitV":{const r=l("axis",e,t,n),a=l("numOrSizeSplits",e,t,n),o=l("x",e,t,n);return s.split(o,a,r)}case"ScatterNd":{const r=l("indices",e,t,n),a=l("values",e,t,n),o=l("shape",e,t,n);return[s.scatterND(r,a,o)]}case"GatherNd":{const r=l("x",e,t,n),a=l("indices",e,t,n);return[s.gatherND(r,a)]}case"SparseToDense":{const r=l("sparseIndices",e,t,n),a=l("outputShape",e,t,n),o=l("sparseValues",e,t,n),i=l("defaultValue",e,t,n);return[s.sparseToDense(r,o,a,o.dtype===i.dtype?i:s.cast(i,o.dtype))]}default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const ww=(e,t,n,s=J)=>{switch(e.op){case"SparseFillEmptyRows":{const{outputIndices:r,outputValues:a,emptyRowIndicator:o,reverseIndexMap:i}=s.sparse.sparseFillEmptyRows(l("indices",e,t,n),l("values",e,t,n),l("denseShape",e,t,n),l("defaultValue",e,t,n));return[r,a,o,i]}case"SparseReshape":{const{outputIndices:r,outputShape:a}=s.sparse.sparseReshape(l("inputIndices",e,t,n),l("inputShape",e,t,n),l("newShape",e,t,n));return[r,a]}case"SparseSegmentMean":return[s.sparse.sparseSegmentMean(l("data",e,t,n),l("indices",e,t,n),l("segmentIds",e,t,n))];case"SparseSegmentSum":return[s.sparse.sparseSegmentSum(l("data",e,t,n),l("indices",e,t,n),l("segmentIds",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const Nw=(e,t,n,s=J)=>{switch(e.op){case"FFT":return[s.fft(l("x",e,t,n))];case"IFFT":return[s.ifft(l("x",e,t,n))];case"RFFT":return[s.rfft(l("x",e,t,n))];case"IRFFT":return[s.irfft(l("x",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const Tw=(e,t,n,s=J)=>{switch(e.op){case"StringNGrams":{const{nGrams:r,nGramsSplits:a}=s.string.stringNGrams(l("data",e,t,n),l("dataSplits",e,t,n),l("separator",e,t,n),l("nGramWidths",e,t,n),l("leftPad",e,t,n),l("rightPad",e,t,n),l("padWidth",e,t,n),l("preserveShortSequences",e,t,n));return[r,a]}case"StringSplit":{const{indices:r,values:a,shape:o}=s.string.stringSplit(l("input",e,t,n),l("delimiter",e,t,n),l("skipEmpty",e,t,n));return[r,a,o]}case"StringToHashBucketFast":return[s.string.stringToHashBucketFast(l("input",e,t,n),l("numBuckets",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */const Sw=(e,t,n,s=J)=>{switch(e.op){case"Cast":return[s.cast(l("x",e,t,n),l("dtype",e,t,n))];case"ExpandDims":{const r=l("axis",e,t,n);return[s.expandDims(l("x",e,t,n),r)]}case"Squeeze":{const r=l("axis",e,t,n);return[s.squeeze(l("x",e,t,n),r)]}case"Reshape":return[s.reshape(l("x",e,t,n),l("shape",e,t,n))];case"MirrorPad":return[s.mirrorPad(l("x",e,t,n),l("padding",e,t,n),l("mode",e,t,n))];case"PadV2":case"Pad":return[s.pad(l("x",e,t,n),l("padding",e,t,n),l("constantValue",e,t,n))];case"SpaceToBatchND":{const r=l("blockShape",e,t,n),a=l("paddings",e,t,n);return[s.spaceToBatchND(l("x",e,t,n),r,a)]}case"BatchToSpaceND":{const r=l("blockShape",e,t,n),a=l("crops",e,t,n);return[s.batchToSpaceND(l("x",e,t,n),r,a)]}case"DepthToSpace":{const r=l("blockSize",e,t,n),a=l("dataFormat",e,t,n).toUpperCase();return[s.depthToSpace(l("x",e,t,n),r,a)]}case"BroadcastTo":return[s.broadcastTo(l("x",e,t,n),l("shape",e,t,n))];case"BroadcastArgs":return[s.broadcastArgs(l("s0",e,t,n),l("s1",e,t,n))];default:throw TypeError(`Node type ${e.op} is not implemented`)}};/**
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
 */function er(e,t,n,s,r=gt){const a=((o,i,u)=>{switch(o.category){case"arithmetic":return r(()=>Zb(o,i,u));case"basic_math":return r(()=>Qb(o,i,u));case"control":return aw(o,i,u);case"convolution":return r(()=>ow(o,i,u));case"creation":return r(()=>iw(o,i,u));case"dynamic":return uw(o,i,u);case"evaluation":return r(()=>cw(o,i,u));case"image":return r(()=>fw(o,i,u));case"graph":return r(()=>lw(o,i,u));case"logical":return r(()=>mw(o,i,u));case"matrices":return r(()=>dw(o,i,u));case"normalization":return r(()=>gw(o,i,u));case"reduction":return r(()=>yw(o,i,u));case"slice_join":return r(()=>bw(o,i,u));case"sparse":return r(()=>ww(o,i,u));case"spectral":return r(()=>Nw(o,i,u));case"string":return r(()=>Tw(o,i,u));case"transformation":return r(()=>Sw(o,i,u));case"hash_table":return hw(o,i,u,s);case"custom":const c=Oa(o.op);if(c&&c.customExecutor)return c.customExecutor(new Jb(o,i,u));throw TypeError(`Custom op ${o.op} is not registered.`);default:throw TypeError(`Unknown op '${o.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(e,t,n);return Mt(a)?a.then(o=>[].concat(o)):[].concat(a)}class nr{constructor(t={},n={},s={},r={}){this.weightMap=t,this.tensorArrayMap=n,this.tensorListMap=s,this.functionMap=r,this.rootContext={id:0,frameName:"",iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(t,n){return{id:t,frameName:n,iterationId:0}}set currentContext(t){this.contexts!==t&&(this.contexts=t,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){const t=[];for(let n=0;n<this.contexts.length-1;n++){const s=this.contexts.slice(0,this.contexts.length-n);t.push(this.contextIdforContexts(s))}t.push(""),this._currentContextIds=t}contextIdforContexts(t){return t?t.map(n=>n.id===0&&n.iterationId===0?"":`${n.frameName}-${n.iterationId}`).join("/"):""}enterFrame(t){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,t)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw new Error("Cannot exit frame, the context is empty")}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;const t=Object.assign({},this.contexts[this.contexts.length-1]);t.iterationId+=1,t.id=this.lastId,this.contexts.splice(-1,1,t),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw new Error("Cannot increase frame iteration, the context is empty")}getWeight(t){return this.weightMap[t]}addTensorArray(t){this.tensorArrayMap[t.id]=t}getTensorArray(t){return this.tensorArrayMap[t]}addTensorList(t){this.tensorListMap[t.id]=t}getTensorList(t){return this.tensorListMap[t]}dispose(t){for(const n in this.tensorArrayMap)this.tensorArrayMap[n].clearAndClose(t);for(const n in this.tensorListMap)this.tensorListMap[n].clearAndClose(t)}}/**
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
 */function sr(e,t,n,s){const r=new Set,a=[];let o=null,i=null;const u=new Set,c=Object.keys(e).map(m=>ut(m)[0]);let p=[];s!=null&&(p=s.map(m=>ut(m.name)[0]));const h=[...t];for(;h.length>0;){const m=h.pop();if((Ba(m)||vw(m)||Iw(m))&&o==null&&(o=m,i=o.children.map(y=>y.name).filter(y=>r.has(y))),r.add(m.name),n[m.name]==null&&c.indexOf(m.name)===-1&&p.indexOf(m.name)===-1){if(m.inputs.length===0){a.push(m.name);continue}m.inputs.forEach(y=>{u.has(y.name)||(u.add(y.name),h.push(y))})}}return{inputs:e,outputs:t,usedNodes:r,missingInputs:a,dynamicNode:o,syncInputs:i}}function kw(e,t,n){const{usedNodes:s,inputs:r}=n,a=[],o=Object.keys(r).map(p=>ut(p)[0]).map(p=>e.nodes[p]),i=e.initNodes;o.forEach(p=>{s.has(p.name)&&a.push(p)}),e.weights.forEach(p=>{s.has(p.name)&&a.push(p)}),i!=null&&i.forEach(p=>{s.has(p.name)&&a.push(p)});const u=new Set,c=[];for(;a.length>0;){const p=a.pop();u.add(p.name),t[p.name]||c.push(p),p.children.forEach(h=>{!u.has(h.name)&&s.has(h.name)&&h.inputs.every(m=>u.has(m.name))&&a.push(h)})}return c}const $w=["Switch","Merge","Enter","Exit","NextIteration","StatelessIf","StatelessWhile","if","While"],Ew=["NonMaxSuppressionV2","NonMaxSuppressionV3","NonMaxSuppressionV5","Where"],_w=["HashTable","HashTableV2","LookupTableImport","LookupTableImportV2","LookupTableFind","LookupTableFindV2","LookupTableSize","LookupTableSizeV2"];function Ba(e){return $w.indexOf(e.op)>=0}function vw(e){return Ew.indexOf(e.op)>=0}function Iw(e){return _w.indexOf(e.op)>=0}/**
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
 */class tn{constructor(t,n){this.graph=t,this.parent=n,this.compiledMap=new Map,this._weightMap={},this.SEPERATOR=",",this._functions={},this._functionExecutorMap={},this.intermediateTensors={},this.keepTensorForDebug=!1,this._outputs=t.outputs,this._inputs=t.inputs,this._initNodes=t.initNodes,this._signature=t.signature,this._functions=t.functions,t.functions!=null&&Object.keys(t.functions).forEach(s=>{this._functionExecutorMap[s]=new tn(t.functions[s],this)})}get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(t){const n=Object.keys(t).map(s=>t[s].map(r=>r.id));this._weightIds=[].concat(...n),this._weightMap=t}set resourceManager(t){this._resourceManager=t}get inputs(){return this._inputs.map(t=>({name:t.name,shape:t.attrParams.shape?t.attrParams.shape.value:void 0,dtype:t.attrParams.dtype?t.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(t=>({name:t.name,shape:t.attrParams.shape?t.attrParams.shape.value:void 0,dtype:t.attrParams.dtype?t.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(t=>t.signatureKey||t.name)}get outputNodes(){return this._outputs.map(t=>{const n=t.signatureKey||t.name;return t.defaultOutput?`${n}:${t.defaultOutput}`:n})}get functions(){return Object.keys(this._functions).reduce((t,n)=>(t[n]=this._functions[n].signature,t),{})}getCompilationKey(t,n){const s=t.map(a=>a.name).sort(),r=n.map(a=>a.name).sort();return s.join(this.SEPERATOR)+"--"+r.join(this.SEPERATOR)}compile(t,n){const s=sr(t,n,this.weightMap,this._initNodes),{missingInputs:r,dynamicNode:a,syncInputs:o}=s;if(a!=null)throw new Error(`This execution contains the node '${a.name}', which has the dynamic op '${a.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${o}]`);if(r.length>0){const i=n.map(c=>c.name),u=Object.keys(t);throw new Error(`Cannot compute the outputs [${i}] from the provided inputs [${u}]. Missing the following inputs: [${r}]`)}return kw(this.graph,this.weightMap,s)}execute(t,n){t=this.mapInputs(t);const s=Object.keys(t).sort();this.checkInputs(t),this.checkInputShapeAndType(t),n=this.mapOutputs(n),this.checkOutputs(n);const r=s.map(h=>this.graph.nodes[ut(h)[0]]),a=n.map(h=>ut(h)[0]);let o=a.map(h=>this.graph.nodes[h]);this.resetIntermediateTensors(),o.length===0&&(o=this._outputs);const i=this.getCompilationKey(r,o);let u=this.compiledMap.get(i);u==null&&(u=this.compile(t,o),this.compiledMap.set(i,u));const c={},p={};return gt(()=>{const h=new nr(this.weightMap,c,p,this.functionExecutorMap),m=Object.assign({},this.weightMap);Object.keys(t).forEach(N=>{const[w,k]=ut(N),v=[];v[k]=t[N],m[w]=v});const y=this.getFrozenTensorIds(m),T={};for(let N=0;N<u.length;N++){const w=u[N];if(!m[w.name]){const k=er(w,m,h,this._resourceManager);if(Mt(k))throw new Error(`The execution of the op '${w.op}' returned a promise. Please use model.executeAsync() instead.`);m[w.name]=k,this.checkTensorForDisposal(w.name,w,m,h,y,a,T)}}return this.parent==null&&h.dispose(y),n.map(N=>nt(N,m,h))})}getFrozenTensorIds(t){const n=[].concat.apply([],Object.keys(t).map(s=>t[s]).map(s=>s.map(r=>r.id)));return new Set(n)}checkTensorForDisposal(t,n,s,r,a,o,i){n.category==="control"||o.indexOf(t)!==-1||(s[t].forEach(u=>{u!=null&&(i[u.id]=(i[u.id]||0)+n.children.length)}),n.inputs.forEach(u=>{if(u.category!=="control"){const c=pb(u.name,s,r);c!=null&&c.forEach(p=>{if(p&&!p.kept&&!a.has(p.id)){const h=i[p.id];if(h===1){if(!this.keepTensorForDebug)p.dispose();else{const[m,y]=wt(n.name,r);this.intermediateTensors[m]?this.intermediateTensors[m][y]=p:(this.intermediateTensors[m]=[],this.intermediateTensors[m][y]=p)}delete i[p.id]}else h!=null&&i[p.id]--}})}}))}async executeAsync(t,n){return this._executeAsync(t,n)}disposeIntermediateTensors(){this.intermediateTensors&&(Object.keys(this.intermediateTensors).forEach(t=>this.intermediateTensors[t].forEach(n=>n.dispose())),this.disposeTensorsMap())}disposeTensorsMap(){this.tensorsMap&&Object.keys(this.tensorsMap).forEach(t=>{this.tensorsMap[t].forEach(s=>{s&&!s.kept&&!s.isDisposed&&!this.keepIds.has(s.id)&&s.dispose()})})}getIntermediateTensors(){return this.tensorsMap}resetIntermediateTensors(){for(const t in this.intermediateTensors)this.intermediateTensors[t].forEach(n=>n.dispose()),delete this.intermediateTensors[t]}async _executeAsync(t,n,s=!1,r={},a={}){s||(t=this.mapInputs(t),this.checkInputs(t),this.checkInputShapeAndType(t),n=this.mapOutputs(n),this.checkOutputs(n));try{this.keepTensorForDebug=x().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(p){console.warn(p.message)}this.resetIntermediateTensors();const o=new nr(this.weightMap,r,a,this.functionExecutorMap);this.tensorsMap=await this.executeWithControlFlow(t,o,n,s);const i=n.map(p=>nt(p,this.tensorsMap,o)),u=i.map(p=>p.id),c=Object.keys(t).map(p=>t[p].id);return this.keepIds=new Set([...u,...c,...this.weightIds]),this.keepTensorForDebug||this.disposeTensorsMap(),this.parent==null&&o.dispose(this.keepIds),i}async executeFunctionAsync(t,n,s){const r=t.reduce((a,o,i)=>(a[this.inputs[i].name]=o,a),{});return this._executeAsync(r,this.outputNodes,!0,n,s)}async executeWithControlFlow(t,n,s,r){const a=Object.keys(t),o=a.map($=>this.graph.nodes[ut($)[0]]),i=s.map($=>ut($)[0]);let u=i.map($=>this.graph.nodes[$]);u.length===0&&(u=this._outputs);const{usedNodes:c,missingInputs:p,dynamicNode:h,syncInputs:m}=sr(t,u,this.weightMap,this._initNodes),y=[...o,...this.graph.weights,...this._initNodes||[]].map($=>({node:$,contexts:n.currentContext})),T=Object.assign({},this.weightMap);Object.keys(t).forEach($=>{const[E,I]=ut($),_=[];_[I]=t[$],T[E]=_});const N={},w=this.getFrozenTensorIds(T),k={};for(;y.length>0;){const $=this.processStack(o,y,n,T,k,w,i,N,c);await Promise.all($)}h==null&&!r&&console.warn("This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.");const v=u.filter($=>!Ba($)&&!nt($.name,T,n)).map($=>$.name);if(v.length>0){let $="";throw h!=null&&($=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${m}]`),new Error(`Cannot compute the outputs [${v}] from the provided inputs [${a}]. Consider providing the following inputs: [${p}]. ${$}`)}return T}processStack(t,n,s,r,a,o,i,u,c){const p=[];for(;n.length>0;){const h=n.pop();s.currentContext=h.contexts;let m="";if(h.node.op==="Enter"&&l("isConstant",h.node,r,s)&&([m]=wt(h.node.name,s)),r[h.node.name]==null){const y=er(h.node,r,s,this._resourceManager);m||([m]=wt(h.node.name,s));const T=s.currentContext;Mt(y)?p.push(y.then(N=>(r[m]=N,s.currentContext=T,this.checkTensorForDisposal(m,h.node,r,s,o,i,u),this.processChildNodes(h.node,n,s,r,a,c),N))):(r[m]=y,this.checkTensorForDisposal(m,h.node,r,s,o,i,u),this.processChildNodes(h.node,n,s,r,a,c))}else this.processChildNodes(h.node,n,s,r,a,c)}return p}processChildNodes(t,n,s,r,a,o){t.children.forEach(i=>{const[u]=wt(i.name,s);a[u]||!o.has(i.name)||(i.op==="Merge"?i.inputNames.some(c=>!!nt(c,r,s))&&(a[u]=!0,n.push({contexts:s.currentContext,node:i})):i.inputNames.every(c=>!!nt(c,r,s))&&(a[u]=!0,n.push({contexts:s.currentContext,node:i})))})}dispose(){Object.keys(this.weightMap).forEach(t=>this.weightMap[t].forEach(n=>n.dispose()))}checkInputShapeAndType(t){Object.keys(t).forEach(n=>{const s=t[n],[r]=ut(n),a=this.graph.nodes[r];if(a.attrParams.shape&&a.attrParams.shape.value){const o=a.attrParams.shape.value,i=o.length===s.shape.length&&s.shape.every((u,c)=>o[c]===-1||o[c]===u);d(i,()=>`The shape of dict['${a.name}'] provided in model.execute(dict) must be [${o}], but was [${s.shape}]`)}a.attrParams.dtype&&a.attrParams.dtype.value&&d(s.dtype===a.attrParams.dtype.value,()=>`The dtype of dict['${a.name}'] provided in model.execute(dict) must be ${a.attrParams.dtype.value}, but was ${s.dtype}`)})}mapInputs(t){const n={};for(const s in t)if(this._signature!=null&&this._signature.inputs!=null&&this._signature.inputs[s]!=null){const r=this._signature.inputs[s];n[r.name]=t[s]}else n[s]=t[s];return n}checkInputs(t){const n=Object.keys(t).filter(s=>{const[r]=ut(s);return this.graph.nodes[r]==null});if(n.length>0)throw new Error(`The dict provided in model.execute(dict) has keys: [${n}] that are not part of graph`)}mapOutputs(t){return t.map(n=>this._signature!=null&&this._signature.outputs!=null&&this._signature.outputs[n]!=null?this._signature.outputs[n].name:n,{})}checkOutputs(t){t.forEach(n=>{const[s]=ut(n);if(!this.graph.nodes[s])throw new Error(`The output '${n}' is not found in the graph`)})}}class xw{constructor(t={},n={}){this.hashTableNameToHandle=t,this.hashTableMap=n}addHashTable(t,n){this.hashTableNameToHandle[t]=n.handle,this.hashTableMap[n.id]=n}getHashTableHandleByName(t){return this.hashTableNameToHandle[t]}getHashTableById(t){return this.hashTableMap[t]}dispose(){for(const t in this.hashTableMap)this.hashTableMap[t].clearAndClose(),delete this.hashTableMap[t];for(const t in this.hashTableNameToHandle)this.hashTableNameToHandle[t].dispose(),delete this.hashTableNameToHandle[t]}}/**
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
 */const Dw="?tfjs-format=file",Aw="model.json";class Pa{constructor(t,n={},s=Wr){this.modelUrl=t,this.loadOptions=n,this.version="n/a",this.io=s,n==null&&(this.loadOptions={}),this.resourceManager=new xw}get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}findIOHandler(){const t=this.modelUrl;if(t.load!=null)this.handler=t;else if(this.loadOptions.requestInit!=null)this.handler=this.io.browserHTTPRequest(t,this.loadOptions);else{const n=this.io.getLoadHandlers(t,this.loadOptions);if(n.length===0)n.push(this.io.browserHTTPRequest(t,this.loadOptions));else if(n.length>1)throw new Error(`Found more than one (${n.length}) load handlers for URL '${[t]}'`);this.handler=n[0]}}load(){if(this.findIOHandler(),this.handler.load==null)throw new Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");const t=this.handler.load();return Mt(t)?t.then(n=>this.loadSync(n)):this.loadSync(t)}loadSync(t){this.artifacts=t;const n=this.artifacts.modelTopology;let s=this.artifacts.signature;if(this.artifacts.userDefinedMetadata!=null){const a=this.artifacts.userDefinedMetadata;a.signature!=null&&(s=a.signature),a.structuredOutputKeys!=null&&(this.structuredOutputKeys=a.structuredOutputKeys)}this.signature=s,this.version=`${n.versions.producer}.${n.versions.minConsumer}`;const r=this.io.decodeWeights(this.artifacts.weightData,this.artifacts.weightSpecs);if(this.executor=new tn(Js.Instance.transformGraph(n,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(r),this.executor.resourceManager=this.resourceManager,t.modelInitializer!=null&&t.modelInitializer.node!=null){const a=Js.Instance.transformGraph(t.modelInitializer);this.initializer=new tn(a),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializer.executeAsync({},[])}return!0}async save(t,n){if(typeof t=="string"){const s=this.io.getSaveHandlers(t);if(s.length===0)throw new Error(`Cannot find any save handlers for URL '${t}'`);if(s.length>1)throw new Error(`Found more than one (${s.length}) save handlers for URL '${t}'`);t=s[0]}if(t.save==null)throw new Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return t.save(this.artifacts)}predict(t,n){const s=this.execute(t,this.outputNodes);if(this.structuredOutputKeys){const r=s instanceof W?[s]:s,a={};return r.forEach((o,i)=>a[this.structuredOutputKeys[i]]=o),a}return s}normalizeInputs(t){if(!(t instanceof W)&&!Array.isArray(t))return t;if(t=Array.isArray(t)?t:[t],t.length!==this.inputNodes.length)throw new Error(`Input tensor count mismatch,the graph model has ${this.inputNodes.length} placeholders, while there are ${t.length} input tensors.`);return this.inputNodes.reduce((n,s,r)=>(n[s]=t[r],n),{})}normalizeOutputs(t){return t=t||this.outputNodes,Array.isArray(t)?t:[t]}execute(t,n){t=this.normalizeInputs(t),n=this.normalizeOutputs(n);const s=this.executor.execute(t,n);return s.length>1?s:s[0]}async executeAsync(t,n){t=this.normalizeInputs(t),n=this.normalizeOutputs(n);const s=await this.executor.executeAsync(t,n);return s.length>1?s:s[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(t){return Object.keys(t).reduce((n,s)=>(n[s]=[t[s]],n),{})}dispose(){this.executor.dispose(),this.initializer&&this.initializer.dispose(),this.resourceManager.dispose()}}async function UN(e,t={},n=Wr){if(e==null)throw new Error("modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model");t==null&&(t={}),t.fromTFHub&&typeof e=="string"&&(e=Ow(e));const s=new Pa(e,t,n);return await s.load(),s}function WN(e){if(e==null)throw new Error("modelUrl in loadGraphModelSync() cannot be null. Please provide model artifacts or an IOHandler that loads the model");let t;if(e instanceof Array){const[s,r]=e;if(!s)throw new Error("modelJSON must be the first element of the array");if(!r||!(r instanceof ArrayBuffer))throw new Error("An ArrayBuffer of weights must be the second element of the array");if(!("modelTopology"in s))throw new Error("Model JSON is missing 'modelTopology'");if(!("weightsManifest"in s))throw new Error("Model JSON is missing 'weightsManifest'");const a=fs(s.weightsManifest),o=ps(s,a,r);t=Ge(o)}else if("load"in e)t=e;else if("modelTopology"in e&&"weightSpecs"in e&&"weightData"in e)t=Ge(e);else throw new Error("Unknown model format");const n=new Pa(t);return n.load(),n}function Ow(e){return e.endsWith("/")||(e=e+"/"),`${e}${Aw}${Dw}`}export{wp as $,ho as A,bn as B,Ce as C,CN as D,BN as E,yo as F,ON as G,AN as H,br as I,pn as J,bo as K,wo as L,_o as M,fp as N,to as O,Io as P,Fe as Q,cu as R,Nu as S,W as T,xo as U,yr as V,Oo as W,Co as X,Po as Y,Lo as Z,Bs as _,gt as a,du as a$,on as a0,zo as a1,Ko as a2,Uo as a3,Ho as a4,Wo as a5,FN as a6,qo as a7,Mo as a8,Yo as a9,Ri as aA,zi as aB,Vi as aC,ji as aD,Wi as aE,qi as aF,Gi as aG,Mi as aH,Yi as aI,Hi as aJ,Ji as aK,Xi as aL,Zi as aM,at as aN,Qi as aO,tu as aP,eu as aQ,nu as aR,au as aS,ti as aT,uu as aU,hu as aV,pu as aW,lu as aX,fu as aY,Ju as aZ,mu as a_,ni as aa,ri as ab,ai as ac,oi as ad,ci as ae,li as af,pi as ag,hi as ah,fi as ai,Ps as aj,Ls as ak,di as al,mi as am,gi as an,yi as ao,ki as ap,$i as aq,Ei as ar,vi as as,xi as at,Di as au,Ai as av,Kw as aw,Ci as ax,Fi as ay,Bi as az,$e as b,ht as b$,gu as b0,Ll as b1,bu as b2,$u as b3,Tu as b4,Du as b5,Iu as b6,Au as b7,Ou as b8,Cu as b9,kN as bA,wN as bB,Le as bC,Yf as bD,ge as bE,$N as bF,hn as bG,ya as bH,tp as bI,B as bJ,g as bK,f as bL,kl as bM,Vn as bN,P as bO,b as bP,Pt as bQ,Et as bR,yt as bS,Oe as bT,en as bU,is as bV,VN as bW,ys as bX,Wn as bY,cn as bZ,bs as b_,Fu as ba,xu as bb,_u as bc,Jw as bd,Pu as be,Xu as bf,Lu as bg,Ru as bh,zu as bi,Vu as bj,ju as bk,vu as bl,Ku as bm,Uu as bn,wr as bo,Wu as bp,qu as bq,Gu as br,Yu as bs,Zu as bt,x as bu,Ka as bv,Fw as bw,fN as bx,_e as by,$n as bz,H as c,ou as c$,Wf as c0,At as c1,Mg as c2,Yg as c3,Xg as c4,Jd as c5,fo as c6,mo as c7,No as c8,To as c9,oo as cA,si as cB,ii as cC,ui as cD,Fs as cE,bi as cF,wi as cG,Ni as cH,Ti as cI,Si as cJ,Oi as cK,qw as cL,_i as cM,Ii as cN,Uw as cO,Ww as cP,Pi as cQ,Gw as cR,Hw as cS,Li as cT,Ki as cU,Ui as cV,xr as cW,Mw as cX,su as cY,ru as cZ,js as c_,So as ca,ko as cb,Eo as cc,$o as cd,vo as ce,Pw as cf,Bw as cg,Do as ch,Ao as ci,Lw as cj,Fo as ck,Bo as cl,Ro as cm,Vo as cn,Rw as co,jo as cp,Go as cq,Xo as cr,Jo as cs,Zo as ct,Qo as cu,Vw as cv,zw as cw,mr as cx,ei as cy,jw as cz,$l as d,$h as d$,iu as d0,ot as d1,Xw as d2,Yw as d3,yu as d4,wu as d5,ku as d6,Su as d7,Eu as d8,Bu as d9,tg as dA,jp as dB,Ke as dC,_N as dD,vN as dE,_t as dF,Wp as dG,Ct as dH,Bt as dI,Mp as dJ,Xp as dK,Zp as dL,th as dM,sh as dN,ln as dO,oh as dP,uh as dQ,hh as dR,sN as dS,mh as dT,gh as dU,Es as dV,bh as dW,Nh as dX,vt as dY,Sh as dZ,pN as d_,xn as da,Hu as db,Mu as dc,Zw as dd,qe as de,Wl as df,Hl as dg,Ml as dh,Xl as di,Zl as dj,np as dk,rp as dl,op as dm,up as dn,lp as dp,hp as dq,Zr as dr,kp as ds,xp as dt,un as du,Fp as dv,Pp as dw,Rp as dx,Qr as dy,ta as dz,jt as e,Je as e$,gs as e0,iN as e1,vh as e2,xh as e3,lN as e4,hN as e5,Fh as e6,Ph as e7,hg as e8,Rh as e9,mg as eA,Wr as eB,Ea as eC,lf as eD,hf as eE,mf as eF,la as eG,yf as eH,ws as eI,ob as eJ,wf as eK,Tf as eL,xe as eM,pa as eN,_f as eO,xf as eP,fa as eQ,Xe as eR,ma as eS,da as eT,Bf as eU,ib as eV,Lf as eW,pe as eX,ga as eY,Vf as eZ,Kf as e_,sa as ea,cN as eb,uN as ec,xa as ed,na as ee,jh as ef,Jh as eg,ef as eh,oa as ei,Ss as ej,TN as ek,SN as el,ia as em,Xr as en,vg as eo,ua as ep,cg as eq,Rs as er,En as es,_n as et,PN as eu,LN as ev,mn as ew,ca as ex,Ze as ey,an as ez,Pl as f,ha as f$,mN as f0,Hf as f1,Un as f2,Jf as f3,Qf as f4,ng as f5,em as f6,sm as f7,$t as f8,fn as f9,ka as fA,NN as fB,Qt as fC,Hm as fD,Mm as fE,Xm as fF,Zm as fG,ks as fH,$a as fI,ed as fJ,rg as fK,IN as fL,Ns as fM,sd as fN,ad as fO,yN as fP,EN as fQ,id as fR,cd as fS,rb as fT,pd as fU,fd as fV,dd as fW,yd as fX,wd as fY,Td as fZ,kd as f_,ba as fa,qt as fb,om as fc,um as fd,pm as fe,fm as ff,dm as fg,ym as fh,Sm as fi,Na as fj,zr as fk,Em as fl,dN as fm,vm as fn,xm as fo,Am as fp,Bm as fq,Ta as fr,Rm as fs,Sa as ft,De as fu,bN as fv,Ie as fw,jm as fx,tN as fy,gn as fz,d as g,eo as g$,wa as g0,ub as g1,ig as g2,sb as g3,Ae as g4,_a as g5,It as g6,va as g7,Fd as g8,cb as g9,Ia as gA,Pa as gB,KN as gC,WN as gD,jN as gE,or as gF,xl as gG,ao as gH,ro as gI,rn as gJ,sn as gK,rc as gL,ir as gM,We as gN,Ja as gO,Se as gP,qa as gQ,ar as gR,$c as gS,Qa as gT,In as gU,Ha as gV,Kt as gW,vr as gX,he as gY,Rt as gZ,DN as g_,j as ga,aN as gb,Pd as gc,Kn as gd,Nt as ge,Hr as gf,Ld as gg,Rd as gh,zd as gi,oN as gj,ke as gk,gN as gl,jd as gm,Ud as gn,qd as go,nN as gp,eN as gq,Gd as gr,ne as gs,kr as gt,Yd as gu,rN as gv,RN as gw,zN as gx,Xd as gy,de as gz,Y as h,mp as h0,xN as h1,nn as h2,Wa as h3,no as h4,Is as h5,ra as h6,bt as h7,Kh as h8,Jr as h9,vs as ha,_s as hb,qr as hc,Qw as hd,xs as he,Ye as hf,Yr as hg,Mr as hh,sc as hi,ab as i,dt as j,R as k,UN as l,D as m,st as n,Zt as o,O as p,rt as q,S as r,$s as s,Bl as t,Hp as u,le as v,G as w,q as x,gr as y,go as z};
