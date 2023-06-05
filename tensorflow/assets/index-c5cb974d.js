function wr(e,t){for(var n=0;n<t.length;n++){const r=t[n];if(typeof r!="string"&&!Array.isArray(r)){for(const s in r)if(s!=="default"&&!(s in e)){const o=Object.getOwnPropertyDescriptor(r,s);o&&Object.defineProperty(e,s,o.get?o:{enumerable:!0,get:()=>r[s]})}}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}function ec(){return!!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia)}/**
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
 */const br=1e-7,Sr=1e-4;class nc{constructor(t,n){this.backend=t,this.dataMover=n,this.data=new WeakMap,this.dataIdsCount=0}get(t){return this.data.has(t)||this.dataMover.moveData(this.backend,t),this.data.get(t)}set(t,n){this.dataIdsCount++,this.data.set(t,n)}has(t){return this.data.has(t)}delete(t){return this.dataIdsCount--,this.data.delete(t)}numDataIds(){return this.dataIdsCount}}class Er{refCount(t){return K("refCount")}incRef(t){return K("incRef")}timerAvailable(){return!0}time(t){return K("time")}read(t){return K("read")}readSync(t){return K("readSync")}readToGPU(t,n){return K("readToGPU")}numDataIds(){return K("numDataIds")}disposeData(t,n){return K("disposeData")}write(t,n,r){return K("write")}move(t,n,r,s,o){return K("move")}createTensorFromGPUData(t,n,r){return K("createTensorFromGPUData")}memory(){return K("memory")}floatPrecision(){return K("floatPrecision")}epsilon(){return this.floatPrecision()===32?br:Sr}dispose(){return K("dispose")}}function K(e){throw new Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
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
 */function rc(e){let t=e.length,n=0;for(;t>0;)n=Math.random()*t|0,t--,he(e,t,n)}function sc(e,t){if(e.length!==t.length)throw new Error(`Array sizes must match to be shuffled together First array length was ${e.length}Second array length was ${t.length}`);let n=e.length,r=0;for(;n>0;)r=Math.random()*n|0,n--,he(e,n,r),he(t,n,r)}function Zt(e,t,n){return Math.max(e,Math.min(t,n))}function oc(e){return e%2===0?e:e+1}function he(e,t,n){const r=e[t];e[t]=e[n],e[n]=r}function ic(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function x(e,t){if(!e)throw new Error(typeof t=="string"?t:t())}function xr(e,t,n=""){x(Ue(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function ac(e){x(e!=null,()=>"The input to the tensor constructor must be a non-null value.")}function V(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function Ue(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function de(e){return e%1===0}function cc(e){const t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function Ht(e,t){return t<=e.length?e:e+" ".repeat(t-e.length)}function lc(e,t=s=>0,n,r){return new Promise((s,o)=>{let i=0;const c=()=>{if(e()){s();return}i++;const a=t(i);if(n!=null&&i>=n){o();return}r!=null?r(c,a):setTimeout(c,a)};c()})}function uc(e,t){let n=1,r=-1;for(let o=0;o<e.length;++o)if(e[o]>=0)n*=e[o];else if(e[o]===-1){if(r!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${r} and dim ${o}`);r=o}else if(e[o]<0)throw Error(`Shapes can not be < 0. Found ${e[o]} at dim ${o}`);if(r===-1){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(n===0)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);const s=e.slice();return s[r]=t/n,s}function Ir(e,t){const n=t.length;return e=e==null?t.map((r,s)=>s):[].concat(e),x(e.every(r=>r>=-n&&r<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),x(e.every(r=>de(r)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(r=>r<0?n+r:r)}function fc(e,t){const n=[],r=[],s=t!=null&&Array.isArray(t)&&t.length===0,o=t==null||s?null:Ir(t,e).sort();let i=0;for(let c=0;c<e.length;++c){if(o!=null){if(o[i]===c&&e[c]!==1)throw new Error(`Can't squeeze axis ${c} since its dim '${e[c]}' is not 1`);(o[i]==null||o[i]>c)&&e[c]===1&&(n.push(e[c]),r.push(c)),o[i]<=c&&i++}e[c]!==1&&(n.push(e[c]),r.push(c))}return{newShape:n,keptDims:r}}function hc(e,t){return xn(e,t)}function xn(e,t){let n=null;if(e==null||e==="float32")n=new Float32Array(t);else if(e==="int32")n=new Int32Array(t);else if(e==="bool")n=new Uint8Array(t);else if(e==="string")n=new Array(t);else throw new Error(`Unknown data type ${e}`);return n}function Ar(e,t){for(let n=0;n<e.length;n++){const r=e[n];if(isNaN(r)||!isFinite(r))throw Error(`A tensor of type ${t} being uploaded contains ${r}.`)}}function kr(e){return e==="bool"||e==="complex64"||e==="float32"||e==="int32"||e==="string"}function dc(e,t){return!(t==="complex64"||t==="float32"&&e!=="complex64"||t==="int32"&&e!=="float32"&&e!=="complex64"||t==="bool"&&e==="bool")}function ge(e){if(e==="float32"||e==="int32")return 4;if(e==="complex64")return 8;if(e==="bool")return 1;throw new Error(`Unknown dtype ${e}`)}function vr(e){if(e==null)return 0;let t=0;return e.forEach(n=>t+=n.length),t}function Ge(e){return typeof e=="string"||e instanceof String}function Tr(e){return typeof e=="boolean"}function $r(e){return typeof e=="number"}function re(e){return Array.isArray(e)?re(e[0]):e instanceof Float32Array?"float32":e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?"int32":$r(e)?"float32":Ge(e)?"string":Tr(e)?"bool":"float32"}function pe(e){return!!(e&&e.constructor&&e.call&&e.apply)}function me(e,t){for(let n=t;n<e;++n)if(e%n===0)return n;return e}function Vt(e){const t=e.length;if(t<2)return[];const n=new Array(t-1);n[t-2]=e[t-1];for(let r=t-3;r>=0;--r)n[r]=n[r+1]*e[r+1];return n}function In(e,t,n,r=!1){const s=new Array;if(t.length===1){const o=t[0]*(r?2:1);for(let i=0;i<o;i++)s[i]=n[e+i]}else{const o=t[0],i=t.slice(1),c=i.reduce((a,l)=>a*l)*(r?2:1);for(let a=0;a<o;a++)s[a]=In(e+a*c,i,n,r)}return s}function Gt(e,t,n=!1){if(e.length===0)return t[0];const r=e.reduce((s,o)=>s*o)*(n?2:1);if(r===0)return[];if(r!==t.length)throw new Error(`[${e}] does not match the input size ${t.length}${n?" for a complex tensor":""}.`);return In(0,e,t,n)}function gc(e,t){if(Array.isArray(e))return e;if(t==="float32")return e instanceof Float32Array?e:new Float32Array(e);if(t==="int32")return e instanceof Int32Array?e:new Int32Array(e);if(t==="bool"||t==="string")return Uint8Array.from(new Int32Array(e));throw new Error(`Unknown dtype ${t}`)}function Mr(e,t){const n=An(e,t);for(let r=0;r<n.length;r++)n[r]=1;return n}function An(e,t){if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool")return new Uint8Array(e);throw new Error(`Unknown data type ${t}`)}function pc(e,t){const n=e.reduce((r,s)=>r*s,1);if(t==null||t==="float32")return Gt(e,new Float32Array(n));if(t==="int32")return Gt(e,new Int32Array(n));if(t==="bool")return Gt(e,new Uint8Array(n));throw new Error(`Unknown data type ${t}`)}function se(e){e.forEach(t=>{x(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function mc(e,t,n){if(t===0)return 0;if(t===1)return e[0];let r=e[e.length-1];for(let s=0;s<e.length-1;++s)r+=n[s]*e[s];return r}function yc(e,t,n){if(t===0)return[];if(t===1)return[e];const r=new Array(t);for(let s=0;s<r.length-1;++s)r[s]=Math.floor(e/n[s]),e-=r[s]*n[s];return r[r.length-1]=e,r}function We(e){return e&&e.then&&typeof e.then=="function"}/**
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
 */const tn="tfjsflags";class Fr{constructor(t){this.global=t,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=Rr,this.populateURLFlags()}setPlatform(t,n){this.platform!=null&&(v().getBool("IS_TEST")||v().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${t}.`)),this.platformName=t,this.platform=n}registerFlag(t,n,r){if(this.flagRegistry[t]={evaluationFn:n,setHook:r},this.urlFlags[t]!=null){const s=this.urlFlags[t];v().getBool("IS_TEST")||v().getBool("PROD")||console.warn(`Setting feature override from URL ${t}: ${s}.`),this.set(t,s)}}async getAsync(t){return t in this.flags?this.flags[t]:(this.flags[t]=await this.evaluateFlag(t),this.flags[t])}get(t){if(t in this.flags)return this.flags[t];const n=this.evaluateFlag(t);if(We(n))throw new Error(`Flag ${t} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[t]=n,this.flags[t]}getNumber(t){return this.get(t)}getBool(t){return this.get(t)}getString(t){return this.get(t)}getFlags(){return this.flags}get features(){return this.flags}set(t,n){if(this.flagRegistry[t]==null)throw new Error(`Cannot set flag ${t} as it has not been registered.`);this.flags[t]=n,this.flagRegistry[t].setHook!=null&&this.flagRegistry[t].setHook(n)}evaluateFlag(t){if(this.flagRegistry[t]==null)throw new Error(`Cannot evaluate flag '${t}': no evaluation function found.`);return this.flagRegistry[t].evaluationFn()}setFlags(t){this.flags=Object.assign({},t)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const t=this.getQueryParams(this.global.location.search);tn in t&&t[tn].split(",").forEach(r=>{const[s,o]=r.split(":");this.urlFlags[s]=Dr(s,o)})}}function Rr(e){const t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(n,...r)=>(Br(t,r[0],r[1]),r.join("="))),t}function Br(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||"")}function Dr(e,t){const n=t.toLowerCase();return n==="true"||n==="false"?n==="true":`${+n}`===n?+n:t}function v(){return kn}let kn=null;function Nr(e){kn=e}/**
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
 */let ie;function vn(){if(ie==null){let e;if(typeof window<"u")e=window;else if(typeof global<"u")e=global;else if(typeof process<"u")e=process;else if(typeof self<"u")e=self;else throw new Error("Could not find a global object");ie=e}return ie}function _r(){const e=vn();return e._tfGlobals==null&&(e._tfGlobals=new Map),e._tfGlobals}function ze(e,t){const n=_r();if(n.has(e))return n.get(e);{const r=t();return n.set(e,r),n.get(e)}}const Or="Abs",wc="Acos",bc="Acosh",Tn="Add",Sc="AddN",Ec="All",xc="Any",Ic="ArgMax",Ac="ArgMin",kc="Asin",vc="Asinh",Tc="Atan",$c="Atanh",Mc="Atan2",Fc="AvgPool",Rc="AvgPoolGrad",Bc="AvgPool3D",Dc="AvgPool3DGrad",Nc="BatchMatMul",_c="BatchToSpaceND",Oc="Bincount",Cc="BroadcastTo",Lc="BroadcastArgs",$n="Cast",Pc="Ceil",Uc="ClipByValue",Cr="Complex",Lr="ComplexAbs",Gc="Concat",Wc="Conv2D",zc="Conv2DBackpropFilter",qc="Conv2DBackpropInput",Vc="Conv3D",jc="Conv3DBackpropFilterV2",Hc="Conv3DBackpropInputV2",Kc="Cos",Xc="Cosh",Zc="Cumprod",Jc="Cumsum",Yc="CropAndResize",Qc="DenseBincount",tl="DepthToSpace",el="DepthwiseConv2dNative",nl="DepthwiseConv2dNativeBackpropFilter",rl="DepthwiseConv2dNativeBackpropInput",sl="Diag",ol="Dilation2D",il="Dilation2DBackpropInput",al="Dilation2DBackpropFilter",Pr="RealDiv",cl="Einsum",Ur="Elu",ll="EluGrad",ul="Erf",fl="Equal",hl="Exp",dl="ExpandDims",gl="Expm1",pl="FFT",Gr="Fill",ml="FlipLeftRight",yl="Floor",Wr="FloorDiv",wl="FusedBatchNorm",bl="GatherV2",Sl="GatherNd",El="Greater",xl="GreaterEqual",Mn="Identity",Il="IFFT",Al="Imag",kl="IsFinite",vl="IsInf",Tl="IsNan",zr="LeakyRelu",$l="Less",Ml="LessEqual",Fl="LinSpace",Rl="Log",Bl="Log1p",Dl="LogicalAnd",Nl="LogicalNot",_l="LogicalOr",Ol="LogSoftmax",Cl="LRN",Ll="LRNGrad",Pl="Max",qr="Maximum",Ul="MaxPool",Gl="MaxPoolGrad",Wl="MaxPool3D",zl="MaxPool3DGrad",ql="MaxPoolWithArgmax",Vl="Mean",jl="Min",Hl="Minimum",Kl="MirrorPad",Xl="Mod",Zl="Multinomial",Vr="Multiply",Jl="Neg",Yl="NotEqual",Ql="NonMaxSuppressionV3",tu="NonMaxSuppressionV4",eu="NonMaxSuppressionV5",nu="OnesLike",ru="OneHot",su="Pack",ou="PadV2",jr="Pow",Hr="Prelu",iu="Prod",au="RaggedGather",cu="RaggedRange",lu="RaggedTensorToTensor",uu="Range",fu="Real",hu="Reciprocal",Kr="Relu",Xr="Reshape",du="ResizeNearestNeighbor",gu="ResizeNearestNeighborGrad",pu="ResizeBilinear",mu="ResizeBilinearGrad",Zr="Relu6",yu="Reverse",wu="Round",bu="Rsqrt",Su="ScatterNd",Eu="TensorScatterUpdate",xu="SearchSorted",Iu="Select",Au="Selu",ku="Slice",vu="Sin",Tu="Sinh",$u="Sign",Jr="Sigmoid",Mu="Softplus",Yr="Sqrt",Qr="Sum",Fu="SpaceToBatchND",Ru="SplitV",Bu="Softmax",Du="SparseFillEmptyRows",Nu="SparseReshape",_u="SparseSegmentMean",Ou="SparseSegmentSum",Cu="SparseToDense",Lu="SquaredDifference",Pu="Square",Uu="StaticRegexReplace",Gu="StridedSlice",Wu="StringNGrams",zu="StringSplit",qu="StringToHashBucketFast",ts="Sub",Vu="Tan",ju="Tanh",es="Tile",Hu="TopK",Ku="Transform",Xu="Transpose",Zu="Unique",Ju="Unpack",Yu="UnsortedSegmentSum",ns="ZerosLike",rs="Step",Qu="FromPixels",tf="RotateWithOffset",ef="_FusedMatMul",nf="FusedConv2D",rf="FusedDepthwiseConv2D";/**
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
 */function ft(...e){v().getBool("IS_TEST")||v().getBool("PROD")||console.warn(...e)}function ss(...e){v().getBool("IS_TEST")||v().getBool("PROD")||console.log(...e)}/**
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
 */const Jt=ze("kernelRegistry",()=>new Map),ye=ze("gradRegistry",()=>new Map);function en(e,t){const n=Fn(e,t);return Jt.get(n)}function nn(e){return ye.get(e)}function rn(e){const t=Jt.entries(),n=[];for(;;){const{done:r,value:s}=t.next();if(r)break;const[o,i]=s,[c]=o.split("_");c===e&&n.push(i)}return n}function sf(e){const{kernelName:t,backendName:n}=e,r=Fn(t,n);Jt.has(r)&&ft(`The kernel '${t}' for backend '${n}' is already registered`),Jt.set(r,e)}function of(e){const{kernelName:t}=e;ye.has(t)&&v().getBool("DEBUG")&&ft(`Overriding the gradient for '${t}'`),ye.set(t,e)}function Fn(e,t){return`${t}_${e}`}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rn(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}var It=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function os(e){if(e.__esModule)return e;var t=e.default;if(typeof t=="function"){var n=function r(){if(this instanceof r){var s=[null];s.push.apply(s,arguments);var o=Function.bind.apply(t,s);return new o}return t.apply(this,arguments)};n.prototype=t.prototype}else n={};return Object.defineProperty(n,"__esModule",{value:!0}),Object.keys(e).forEach(function(r){var s=Object.getOwnPropertyDescriptor(e,r);Object.defineProperty(n,r,s.get?s:{enumerable:!0,get:function(){return e[r]}})}),n}var we=N,tt=null;try{tt=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function N(e,t,n){this.low=e|0,this.high=t|0,this.unsigned=!!n}N.prototype.__isLong__;Object.defineProperty(N.prototype,"__isLong__",{value:!0});function Z(e){return(e&&e.__isLong__)===!0}N.isLong=Z;var sn={},on={};function At(e,t){var n,r,s;return t?(e>>>=0,(s=0<=e&&e<256)&&(r=on[e],r)?r:(n=_(e,(e|0)<0?-1:0,!0),s&&(on[e]=n),n)):(e|=0,(s=-128<=e&&e<128)&&(r=sn[e],r)?r:(n=_(e,e<0?-1:0,!1),s&&(sn[e]=n),n))}N.fromInt=At;function et(e,t){if(isNaN(e))return t?yt:nt;if(t){if(e<0)return yt;if(e>=Bn)return _n}else{if(e<=-cn)return X;if(e+1>=cn)return Nn}return e<0?et(-e,t).neg():_(e%Nt|0,e/Nt|0,t)}N.fromNumber=et;function _(e,t,n){return new N(e,t,n)}N.fromBits=_;var Yt=Math.pow;function qe(e,t,n){if(e.length===0)throw Error("empty string");if(e==="NaN"||e==="Infinity"||e==="+Infinity"||e==="-Infinity")return nt;if(typeof t=="number"?(n=t,t=!1):t=!!t,n=n||10,n<2||36<n)throw RangeError("radix");var r;if((r=e.indexOf("-"))>0)throw Error("interior hyphen");if(r===0)return qe(e.substring(1),t,n).neg();for(var s=et(Yt(n,8)),o=nt,i=0;i<e.length;i+=8){var c=Math.min(8,e.length-i),a=parseInt(e.substring(i,i+c),n);if(c<8){var l=et(Yt(n,c));o=o.mul(l).add(et(a))}else o=o.mul(s),o=o.add(et(a))}return o.unsigned=t,o}N.fromString=qe;function ot(e,t){return typeof e=="number"?et(e,t):typeof e=="string"?qe(e,t):_(e.low,e.high,typeof t=="boolean"?t:e.unsigned)}N.fromValue=ot;var an=1<<16,is=1<<24,Nt=an*an,Bn=Nt*Nt,cn=Bn/2,ln=At(is),nt=At(0);N.ZERO=nt;var yt=At(0,!0);N.UZERO=yt;var Ft=At(1);N.ONE=Ft;var Dn=At(1,!0);N.UONE=Dn;var be=At(-1);N.NEG_ONE=be;var Nn=_(-1,2147483647,!1);N.MAX_VALUE=Nn;var _n=_(-1,-1,!0);N.MAX_UNSIGNED_VALUE=_n;var X=_(0,-2147483648,!1);N.MIN_VALUE=X;var y=N.prototype;y.toInt=function(){return this.unsigned?this.low>>>0:this.low};y.toNumber=function(){return this.unsigned?(this.high>>>0)*Nt+(this.low>>>0):this.high*Nt+(this.low>>>0)};y.toString=function(t){if(t=t||10,t<2||36<t)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative())if(this.eq(X)){var n=et(t),r=this.div(n),s=r.mul(n).sub(this);return r.toString(t)+s.toInt().toString(t)}else return"-"+this.neg().toString(t);for(var o=et(Yt(t,6),this.unsigned),i=this,c="";;){var a=i.div(o),l=i.sub(a.mul(o)).toInt()>>>0,h=l.toString(t);if(i=a,i.isZero())return h+c;for(;h.length<6;)h="0"+h;c=""+h+c}};y.getHighBits=function(){return this.high};y.getHighBitsUnsigned=function(){return this.high>>>0};y.getLowBits=function(){return this.low};y.getLowBitsUnsigned=function(){return this.low>>>0};y.getNumBitsAbs=function(){if(this.isNegative())return this.eq(X)?64:this.neg().getNumBitsAbs();for(var t=this.high!=0?this.high:this.low,n=31;n>0&&!(t&1<<n);n--);return this.high!=0?n+33:n+1};y.isZero=function(){return this.high===0&&this.low===0};y.eqz=y.isZero;y.isNegative=function(){return!this.unsigned&&this.high<0};y.isPositive=function(){return this.unsigned||this.high>=0};y.isOdd=function(){return(this.low&1)===1};y.isEven=function(){return(this.low&1)===0};y.equals=function(t){return Z(t)||(t=ot(t)),this.unsigned!==t.unsigned&&this.high>>>31===1&&t.high>>>31===1?!1:this.high===t.high&&this.low===t.low};y.eq=y.equals;y.notEquals=function(t){return!this.eq(t)};y.neq=y.notEquals;y.ne=y.notEquals;y.lessThan=function(t){return this.comp(t)<0};y.lt=y.lessThan;y.lessThanOrEqual=function(t){return this.comp(t)<=0};y.lte=y.lessThanOrEqual;y.le=y.lessThanOrEqual;y.greaterThan=function(t){return this.comp(t)>0};y.gt=y.greaterThan;y.greaterThanOrEqual=function(t){return this.comp(t)>=0};y.gte=y.greaterThanOrEqual;y.ge=y.greaterThanOrEqual;y.compare=function(t){if(Z(t)||(t=ot(t)),this.eq(t))return 0;var n=this.isNegative(),r=t.isNegative();return n&&!r?-1:!n&&r?1:this.unsigned?t.high>>>0>this.high>>>0||t.high===this.high&&t.low>>>0>this.low>>>0?-1:1:this.sub(t).isNegative()?-1:1};y.comp=y.compare;y.negate=function(){return!this.unsigned&&this.eq(X)?X:this.not().add(Ft)};y.neg=y.negate;y.add=function(t){Z(t)||(t=ot(t));var n=this.high>>>16,r=this.high&65535,s=this.low>>>16,o=this.low&65535,i=t.high>>>16,c=t.high&65535,a=t.low>>>16,l=t.low&65535,h=0,u=0,f=0,d=0;return d+=o+l,f+=d>>>16,d&=65535,f+=s+a,u+=f>>>16,f&=65535,u+=r+c,h+=u>>>16,u&=65535,h+=n+i,h&=65535,_(f<<16|d,h<<16|u,this.unsigned)};y.subtract=function(t){return Z(t)||(t=ot(t)),this.add(t.neg())};y.sub=y.subtract;y.multiply=function(t){if(this.isZero())return nt;if(Z(t)||(t=ot(t)),tt){var n=tt.mul(this.low,this.high,t.low,t.high);return _(n,tt.get_high(),this.unsigned)}if(t.isZero())return nt;if(this.eq(X))return t.isOdd()?X:nt;if(t.eq(X))return this.isOdd()?X:nt;if(this.isNegative())return t.isNegative()?this.neg().mul(t.neg()):this.neg().mul(t).neg();if(t.isNegative())return this.mul(t.neg()).neg();if(this.lt(ln)&&t.lt(ln))return et(this.toNumber()*t.toNumber(),this.unsigned);var r=this.high>>>16,s=this.high&65535,o=this.low>>>16,i=this.low&65535,c=t.high>>>16,a=t.high&65535,l=t.low>>>16,h=t.low&65535,u=0,f=0,d=0,g=0;return g+=i*h,d+=g>>>16,g&=65535,d+=o*h,f+=d>>>16,d&=65535,d+=i*l,f+=d>>>16,d&=65535,f+=s*h,u+=f>>>16,f&=65535,f+=o*l,u+=f>>>16,f&=65535,f+=i*a,u+=f>>>16,f&=65535,u+=r*h+s*l+o*a+i*c,u&=65535,_(d<<16|g,u<<16|f,this.unsigned)};y.mul=y.multiply;y.divide=function(t){if(Z(t)||(t=ot(t)),t.isZero())throw Error("division by zero");if(tt){if(!this.unsigned&&this.high===-2147483648&&t.low===-1&&t.high===-1)return this;var n=(this.unsigned?tt.div_u:tt.div_s)(this.low,this.high,t.low,t.high);return _(n,tt.get_high(),this.unsigned)}if(this.isZero())return this.unsigned?yt:nt;var r,s,o;if(this.unsigned){if(t.unsigned||(t=t.toUnsigned()),t.gt(this))return yt;if(t.gt(this.shru(1)))return Dn;o=yt}else{if(this.eq(X)){if(t.eq(Ft)||t.eq(be))return X;if(t.eq(X))return Ft;var i=this.shr(1);return r=i.div(t).shl(1),r.eq(nt)?t.isNegative()?Ft:be:(s=this.sub(t.mul(r)),o=r.add(s.div(t)),o)}else if(t.eq(X))return this.unsigned?yt:nt;if(this.isNegative())return t.isNegative()?this.neg().div(t.neg()):this.neg().div(t).neg();if(t.isNegative())return this.div(t.neg()).neg();o=nt}for(s=this;s.gte(t);){r=Math.max(1,Math.floor(s.toNumber()/t.toNumber()));for(var c=Math.ceil(Math.log(r)/Math.LN2),a=c<=48?1:Yt(2,c-48),l=et(r),h=l.mul(t);h.isNegative()||h.gt(s);)r-=a,l=et(r,this.unsigned),h=l.mul(t);l.isZero()&&(l=Ft),o=o.add(l),s=s.sub(h)}return o};y.div=y.divide;y.modulo=function(t){if(Z(t)||(t=ot(t)),tt){var n=(this.unsigned?tt.rem_u:tt.rem_s)(this.low,this.high,t.low,t.high);return _(n,tt.get_high(),this.unsigned)}return this.sub(this.div(t).mul(t))};y.mod=y.modulo;y.rem=y.modulo;y.not=function(){return _(~this.low,~this.high,this.unsigned)};y.and=function(t){return Z(t)||(t=ot(t)),_(this.low&t.low,this.high&t.high,this.unsigned)};y.or=function(t){return Z(t)||(t=ot(t)),_(this.low|t.low,this.high|t.high,this.unsigned)};y.xor=function(t){return Z(t)||(t=ot(t)),_(this.low^t.low,this.high^t.high,this.unsigned)};y.shiftLeft=function(t){return Z(t)&&(t=t.toInt()),(t&=63)===0?this:t<32?_(this.low<<t,this.high<<t|this.low>>>32-t,this.unsigned):_(0,this.low<<t-32,this.unsigned)};y.shl=y.shiftLeft;y.shiftRight=function(t){return Z(t)&&(t=t.toInt()),(t&=63)===0?this:t<32?_(this.low>>>t|this.high<<32-t,this.high>>t,this.unsigned):_(this.high>>t-32,this.high>=0?0:-1,this.unsigned)};y.shr=y.shiftRight;y.shiftRightUnsigned=function(t){if(Z(t)&&(t=t.toInt()),t&=63,t===0)return this;var n=this.high;if(t<32){var r=this.low;return _(r>>>t|n<<32-t,n>>>t,this.unsigned)}else return t===32?_(n,0,this.unsigned):_(n>>>t-32,0,this.unsigned)};y.shru=y.shiftRightUnsigned;y.shr_u=y.shiftRightUnsigned;y.toSigned=function(){return this.unsigned?_(this.low,this.high,!1):this};y.toUnsigned=function(){return this.unsigned?this:_(this.low,this.high,!0)};y.toBytes=function(t){return t?this.toBytesLE():this.toBytesBE()};y.toBytesLE=function(){var t=this.high,n=this.low;return[n&255,n>>>8&255,n>>>16&255,n>>>24,t&255,t>>>8&255,t>>>16&255,t>>>24]};y.toBytesBE=function(){var t=this.high,n=this.low;return[t>>>24,t>>>16&255,t>>>8&255,t&255,n>>>24,n>>>16&255,n>>>8&255,n&255]};N.fromBytes=function(t,n,r){return r?N.fromBytesLE(t,n):N.fromBytesBE(t,n)};N.fromBytesLE=function(t,n){return new N(t[0]|t[1]<<8|t[2]<<16|t[3]<<24,t[4]|t[5]<<8|t[6]<<16|t[7]<<24,n)};N.fromBytesBE=function(t,n){return new N(t[4]<<24|t[5]<<16|t[6]<<8|t[7],t[0]<<24|t[1]<<16|t[2]<<8|t[3],n)};const as=wr({__proto__:null,default:we},[we]);/**
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
 */const pt=we||as;function oe(e){return pt.fromString(e,!0,16)}const On=oe("c3a5c85c97cb3127"),gt=oe("b492b66fbe98f273"),q=oe("9ae16a3b2f90404f");function Se(e){return e.xor(e.shru(47))}function Cn(e,t,n){const r=e.slice(t,t+n);return pt.fromBytes(Array.from(r),!0,!0)}function D(e,t){return Cn(e,t,8)}function un(e,t){return Cn(e,t,4)}function P(e,t){return t===0?e:e.shru(t).or(e.shl(64-t))}function dt(e,t,n=oe("9ddfea08eb382d69")){let r=e.xor(t).mul(n);r=r.xor(r.shru(47));let s=t.xor(r).mul(n);return s=s.xor(s.shru(47)),s=s.mul(n),s}function cs(e,t,n,r,s,o){s=s.add(e),o=P(o.add(s).add(r),21);const i=s;return s=s.add(t),s=s.add(n),o=o.add(P(s,44)),[s.add(r),o.add(i)]}function jt(e,t,n,r){return cs(D(e,t),D(e,t+8),D(e,t+16),D(e,t+24),n,r)}function ls(e,t=e.length){if(t>=8){const n=q.add(t*2),r=D(e,0).add(q),s=D(e,t-8),o=P(s,37).mul(n).add(r),i=P(r,25).add(s).mul(n);return dt(o,i,n)}if(t>=4){const n=q.add(t*2),r=un(e,0);return dt(r.shl(3).add(t),un(e,t-4),n)}if(t>0){const n=e[0],r=e[t>>1],s=e[t-1],o=n+(r<<8),i=t+(s<<2);return Se(q.mul(o).xor(On.mul(i))).mul(q)}return q}function us(e,t=e.length){const n=q.add(t*2),r=D(e,0).mul(gt),s=D(e,8),o=D(e,t-8).mul(n),i=D(e,t-16).mul(q);return dt(P(r.add(s),43).add(P(o,30)).add(i),r.add(P(s.add(q),18)).add(o),n)}function fs(e,t=e.length){const n=q.add(t*2),r=D(e,0).mul(q),s=D(e,8),o=D(e,t-8).mul(n),i=D(e,t-16).mul(q),c=P(r.add(s),43).add(P(o,30)).add(i),a=dt(c,r.add(P(s.add(q),18)).add(o),n),l=D(e,16).mul(n),h=D(e,24),u=c.add(D(e,t-32)).mul(n),f=a.add(D(e,t-24)).mul(n);return dt(P(l.add(h),43).add(P(u,30)).add(f),l.add(P(h.add(r),18)).add(u),n)}function af(e,t=e.length){const n=pt.fromNumber(81,!0);if(t<=32)return t<=16?ls(e,t):us(e,t);if(t<=64)return fs(e,t);let r=n,s=n.mul(gt).add(113),o=Se(s.mul(q).add(113)).mul(q),i=[pt.UZERO,pt.UZERO],c=[pt.UZERO,pt.UZERO];r=r.mul(q).add(D(e,0));let a=0;const l=(t-1>>6)*64,h=l+(t-1&63)-63;do r=P(r.add(s).add(i[0]).add(D(e,a+8)),37).mul(gt),s=P(s.add(i[1]).add(D(e,a+48)),42).mul(gt),r=r.xor(c[1]),s=s.add(i[0]).add(D(e,a+40)),o=P(o.add(c[0]),33).mul(gt),i=jt(e,a,i[1].mul(gt),r.add(c[0])),c=jt(e,a+32,o.add(c[1]),s.add(D(e,a+16))),[o,r]=[r,o],a+=64;while(a!==l);const u=gt.add(o.and(255).shl(1));return a=h,c[0]=c[0].add(t-1&63),i[0]=i[0].add(c[0]),c[0]=c[0].add(i[0]),r=P(r.add(s).add(i[0]).add(D(e,a+8)),37).mul(u),s=P(s.add(i[1]).add(D(e,a+48)),42).mul(u),r=r.xor(c[1].mul(9)),s=s.add(i[0].mul(9).add(D(e,a+40))),o=P(o.add(c[0]),33).mul(u),i=jt(e,a,i[1].mul(u),r.add(c[0])),c=jt(e,a+32,o.add(c[1]),s.add(D(e,a+16))),[o,r]=[r,o],dt(dt(i[0],c[0],u).add(Se(s).mul(On)).add(o),dt(i[1],c[1],u).add(r),u)}/**
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
 */function cf(e,t){return t==="string"?je(e):Ve([e],t)}function hs(e,t){return e instanceof Float32Array&&t==="float32"||e instanceof Int32Array&&t==="int32"||e instanceof Uint8Array&&t==="bool"}function Ve(e,t){if(t==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(e)&&(e=Wt(e)),v().getBool("DEBUG")&&Ar(e,t),hs(e,t))return e;if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool"){const n=new Uint8Array(e.length);for(let r=0;r<n.length;++r)Math.round(e[r])!==0&&(n[r]=1);return n}else throw new Error(`Unknown data type ${t}`)}function Qt(){return v().platform.now()}function je(e,t="utf-8"){return t=t||"utf-8",v().platform.encode(e,t)}function Ee(e,t="utf-8"){return t=t||"utf-8",v().platform.decode(e,t)}function st(e){return v().platform.isTypedArray!=null?v().platform.isTypedArray(e):Rn(e)}function Wt(e,t=[],n=!1){if(t==null&&(t=[]),typeof e=="boolean"||typeof e=="number"||typeof e=="string"||We(e)||e==null||st(e)&&n)t.push(e);else if(Array.isArray(e)||st(e))for(let r=0;r<e.length;++r)Wt(e[r],t,n);else{let r=-1;for(const s of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(s)&&(r=Math.max(r,Number(s)));for(let s=0;s<=r;s++)Wt(e[s],t,n)}return t}/**
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
 */class ds{constructor(t,n){this.backendTimer=t,this.logger=n,n==null&&(this.logger=new ps)}profileKernel(t,n,r){let s;const o=()=>{s=r()};let i;const c=Qt();if(this.backendTimer.timerAvailable())i=this.backendTimer.time(o);else{o();for(const l of s)l.dataSync();i=Promise.resolve({kernelMs:Qt()-c})}if(v().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let l=0;l<s.length;l++){const h=s[l];h.data().then(u=>{gs(u,h.dtype,t)})}return{kernelName:t,outputs:s,inputs:n,timeMs:i.then(l=>l.kernelMs),extraInfo:i.then(l=>l.getExtraProfileInfo!=null?l.getExtraProfileInfo():"")}}logKernelProfile(t){const{kernelName:n,outputs:r,timeMs:s,inputs:o,extraInfo:i}=t;r.forEach(c=>{Promise.all([c.data(),s,i]).then(a=>{this.logger.logKernelProfile(n,c,a[0],a[1],o,a[2])})})}}function gs(e,t,n){if(t!=="float32")return!1;for(let r=0;r<e.length;r++){const s=e[r];if(isNaN(s)||!isFinite(s))return console.warn(`Found ${s} in the result of '${n}'`),!0}return!1}class ps{logKernelProfile(t,n,r,s,o,i){const c=typeof s=="number"?Ht(`${s}ms`,9):s.error,a=Ht(t,25),l=n.rank,h=n.size,u=Ht(n.shape.toString(),14);let f="";for(const d in o){const g=o[d];if(g!=null){const p=g.shape||n.shape,m=p.length;f+=`${d}: ${m}D ${m>0?p:""} `}}console.log(`%c${a}	%c${c}	%c${l}D ${u}	%c${h}	%c${f}	%c${i}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
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
 */function ms(e,t,n){const r={},s={};for(let a=0;a<t.length;a++)r[t[a].id]=!0;for(let a=0;a<e.length;a++){const l=e[a],h=l.inputs;for(const u in h){const f=h[u];let d=!1;for(let g=0;g<t.length;g++)if(r[f.id]){l.outputs.forEach(p=>r[p.id]=!0),d=!0,s[l.id]=!0;break}if(d)break}}const o={};o[n.id]=!0;const i={};for(let a=e.length-1;a>=0;a--){const l=e[a],h=l.inputs;for(let u=0;u<l.outputs.length;u++)if(o[l.outputs[u].id]){for(const f in h)o[h[f].id]=!0,i[l.id]=!0;break}}const c=[];for(let a=0;a<e.length;a++){const l=e[a];if(s[l.id]&&i[l.id]){const h={};for(const f in l.inputs){const d=l.inputs[f];r[d.id]&&(h[f]=d)}const u=Object.assign({},l);u.inputs=h,u.outputs=l.outputs,c.push(u)}}return c}function ys(e,t,n,r){for(let s=t.length-1;s>=0;s--){const o=t[s],i=[];if(o.outputs.forEach(a=>{const l=e[a.id];l!=null?i.push(l):i.push(null)}),o.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${o.kernelName}.`);const c=o.gradient(i);for(const a in o.inputs){if(!(a in c))throw new Error(`Cannot backprop through input ${a}. Available gradients found: ${Object.keys(c)}.`);const l=n(()=>c[a]());if(l.dtype!=="float32")throw new Error(`Error in gradient for op ${o.kernelName}. The gradient of input ${a} must have 'float32' dtype, but has '${l.dtype}'`);const h=o.inputs[a];if(!Ue(l.shape,h.shape))throw new Error(`Error in gradient for op ${o.kernelName}. The gradient of input '${a}' has shape '${l.shape}', which does not match the shape of the input '${h.shape}'`);if(e[h.id]==null)e[h.id]=l;else{const u=e[h.id];e[h.id]=r(u,l),u.dispose()}}}}/**
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
 */const fn=20,Lt=3,ae=7;function ws(e,t,n,r){const s=Vt(t),o=bs(e,t,n,s),i=t.length,c=Kt(e,t,n,s,o),a=["Tensor"];return r&&(a.push(`  dtype: ${n}`),a.push(`  rank: ${i}`),a.push(`  shape: [${t}]`),a.push("  values:")),a.push(c.map(l=>"    "+l).join(`
`)),a.join(`
`)}function bs(e,t,n,r){const s=V(t),o=r[r.length-1],i=new Array(o).fill(0),c=t.length,a=n==="complex64"?Ut(e):e;if(c>1)for(let l=0;l<s/o;l++){const h=l*o;for(let u=0;u<o;u++)i[u]=Math.max(i[u],Pt(a[h+u],0,n).length)}return i}function Pt(e,t,n){let r;return Array.isArray(e)?r=`${parseFloat(e[0].toFixed(ae))} + ${parseFloat(e[1].toFixed(ae))}j`:Ge(e)?r=`'${e}'`:n==="bool"?r=Ln(e):r=parseFloat(e.toFixed(ae)).toString(),Ht(r,t)}function Ln(e){return e===0?"false":"true"}function Kt(e,t,n,r,s,o=!0){const i=n==="complex64"?2:1,c=t[0],a=t.length;if(a===0){if(n==="complex64"){const p=Ut(e);return[Pt(p[0],0,n)]}return n==="bool"?[Ln(e[0])]:[e[0].toString()]}if(a===1){if(c>fn){const m=Lt*i;let b=Array.from(e.slice(0,m)),R=Array.from(e.slice((c-Lt)*i,c*i));return n==="complex64"&&(b=Ut(b),R=Ut(R)),["["+b.map((w,S)=>Pt(w,s[S],n)).join(", ")+", ..., "+R.map((w,S)=>Pt(w,s[c-Lt+S],n)).join(", ")+"]"]}return["["+(n==="complex64"?Ut(e):Array.from(e)).map((m,b)=>Pt(m,s[b],n)).join(", ")+"]"]}const l=t.slice(1),h=r.slice(1),u=r[0]*i,f=[];if(c>fn){for(let p=0;p<Lt;p++){const m=p*u,b=m+u;f.push(...Kt(e.slice(m,b),l,n,h,s,!1))}f.push("...");for(let p=c-Lt;p<c;p++){const m=p*u,b=m+u;f.push(...Kt(e.slice(m,b),l,n,h,s,p===c-1))}}else for(let p=0;p<c;p++){const m=p*u,b=m+u;f.push(...Kt(e.slice(m,b),l,n,h,s,p===c-1))}const d=a===2?",":"";f[0]="["+(c>0?f[0]+d:"");for(let p=1;p<f.length-1;p++)f[p]=" "+f[p]+d;let g=`,
`;for(let p=2;p<a;p++)g+=`
`;return f[f.length-1]=" "+f[f.length-1]+"]"+(o?"":g),f}function Ut(e){const t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}/**
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
 */class Ss{constructor(t,n,r){if(this.dtype=n,this.shape=t.slice(),this.size=V(t),r!=null){const s=r.length;x(s===this.size,()=>`Length of values '${s}' does not match the size inferred by the shape '${this.size}'.`)}if(n==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=r||xn(n,this.size),this.strides=Vt(t)}set(t,...n){n.length===0&&(n=[0]),x(n.length===this.rank,()=>`The number of provided coordinates (${n.length}) must match the rank (${this.rank})`);const r=this.locToIndex(n);this.values[r]=t}get(...t){t.length===0&&(t=[0]);let n=0;for(const s of t){if(s<0||s>=this.shape[n]){const o=`Requested out of range element at ${t}.   Buffer shape=${this.shape}`;throw new Error(o)}n++}let r=t[t.length-1];for(let s=0;s<t.length-1;++s)r+=this.strides[s]*t[s];return this.values[r]}locToIndex(t){if(this.rank===0)return 0;if(this.rank===1)return t[0];let n=t[t.length-1];for(let r=0;r<t.length-1;++r)n+=this.strides[r]*t[r];return n}indexToLoc(t){if(this.rank===0)return[];if(this.rank===1)return[t];const n=new Array(this.shape.length);for(let r=0;r<n.length-1;++r)n[r]=Math.floor(t/this.strides[r]),t-=n[r]*this.strides[r];return n[n.length-1]=t,n}get rank(){return this.shape.length}toTensor(){return rt().makeTensor(this.values,this.shape,this.dtype)}}let rt=null,$t=null;function Es(e){rt=e}function xs(e){$t=e}class Q{constructor(t,n,r,s){this.kept=!1,this.isDisposedInternal=!1,this.shape=t.slice(),this.dtype=n||"float32",this.size=V(t),this.strides=Vt(t),this.dataId=r,this.id=s,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const t=await this.data();return $t.buffer(this.shape,this.dtype,t)}bufferSync(){return $t.buffer(this.shape,this.dtype,this.dataSync())}async array(){const t=await this.data();return Gt(this.shape,t,this.dtype==="complex64")}arraySync(){return Gt(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const t=rt().read(this.dataId);if(this.dtype==="string"){const n=await t;try{return n.map(r=>Ee(r))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return t}dataToGPU(t){return this.throwIfDisposed(),rt().readToGPU(this.dataId,t)}dataSync(){this.throwIfDisposed();const t=rt().readSync(this.dataId);if(this.dtype==="string")try{return t.map(n=>Ee(n))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return t}async bytes(){this.throwIfDisposed();const t=await rt().read(this.dataId);return this.dtype==="string"?t:new Uint8Array(t.buffer)}dispose(){this.isDisposed||(rt().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(t=!1){return $t.print(this,t)}clone(){return this.throwIfDisposed(),$t.clone(this)}toString(t=!1){const n=this.dataSync();return ws(n,this.shape,this.dtype,t)}cast(t){return this.throwIfDisposed(),$t.cast(this,t)}variable(t=!0,n,r){return this.throwIfDisposed(),rt().makeVariable(this,t,n,r)}}Object.defineProperty(Q,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function Is(){return ze("Tensor",()=>Q)}Is();class te extends Q{constructor(t,n,r,s){super(t.shape,t.dtype,t.dataId,s),this.trainable=n,this.name=r}assign(t){if(t.dtype!==this.dtype)throw new Error(`dtype of the new value (${t.dtype}) and previous value (${this.dtype}) must match`);if(!Ue(t.shape,this.shape))throw new Error(`shape of the new value (${t.shape}) and previous value (${this.shape}) must match`);rt().disposeTensor(this),this.dataId=t.dataId,rt().incRef(this,null)}dispose(){rt().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(te,Symbol.hasInstance,{value:e=>e instanceof Q&&e.assign!=null&&e.assign instanceof Function});/**
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
 */var hn;(function(e){e.R0="R0",e.R1="R1",e.R2="R2",e.R3="R3",e.R4="R4",e.R5="R5",e.R6="R6"})(hn||(hn={}));var xe;(function(e){e.float32="float32",e.int32="int32",e.bool="int32",e.complex64="complex64"})(xe||(xe={}));var Ie;(function(e){e.float32="float32",e.int32="int32",e.bool="bool",e.complex64="complex64"})(Ie||(Ie={}));var Ae;(function(e){e.float32="float32",e.int32="float32",e.bool="float32",e.complex64="complex64"})(Ae||(Ae={}));var ke;(function(e){e.float32="complex64",e.int32="complex64",e.bool="complex64",e.complex64="complex64"})(ke||(ke={}));const As={float32:Ae,int32:xe,bool:Ie,complex64:ke};function He(e,t){if(e==="string"||t==="string"){if(e==="string"&&t==="string")return"string";throw new Error(`Can not upcast ${e} with ${t}`)}return As[e][t]}function lf(e){return He(e,"int32")}function Pn(e){return e!=null&&typeof e=="object"&&"texture"in e&&e.texture instanceof WebGLTexture}function Un(e){return typeof GPUBuffer<"u"&&e!=null&&typeof e=="object"&&"buffer"in e&&e.buffer instanceof GPUBuffer}/**
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
 */function kt(e,t){if(e.dtype===t.dtype)return[e,t];const n=He(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function uf(e,t){x(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function Gn(e){const t=[];return Wn(e,t,new Set),t}function Wn(e,t,n){if(e==null)return;if(e instanceof Q){t.push(e);return}if(!ks(e))return;const r=e;for(const s in r){const o=r[s];n.has(o)||(n.add(o),Wn(o,t,n))}}function ks(e){return Array.isArray(e)||typeof e=="object"}/**
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
 */function ce(e){return e.kernelName!=null}class dn{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(t=>t.name)))}}}dispose(){for(const t in this.registeredVariables)this.registeredVariables[t].dispose()}}class _t{constructor(t){this.ENV=t,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new dn}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const r=t[n];if(await this.initializeBackend(r).success){await this.setBackend(r);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:t,asyncInit:n}=this.initializeBackendsAndReturnBest();if(n)throw new Error(`The highest priority backend '${t}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(t)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(t){if(!(t in this.registry))if(t in this.registryFactory){const{asyncInit:n}=this.initializeBackend(t);if(n)return null}else return null;return this.registry[t]}findBackendFactory(t){return t in this.registryFactory?this.registryFactory[t].factory:null}registerBackend(t,n,r=1){return t in this.registryFactory?(ft(`${t} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[t]={factory:n,priority:r},!0)}async setBackend(t){if(this.registryFactory[t]==null)throw new Error(`Backend name '${t}' not found in registry`);if(this.backendName=t,this.registry[t]==null){this.backendInstance=null;const{success:n,asyncInit:r}=this.initializeBackend(t);if(!(r?await n:n))return!1}return this.backendInstance=this.registry[t],this.setupRegisteredKernels(),this.profiler=new ds(this.backendInstance),!0}setupRegisteredKernels(){rn(this.backendName).forEach(n=>{n.setupFunc!=null&&n.setupFunc(this.backendInstance)})}disposeRegisteredKernels(t){rn(t).forEach(r=>{r.disposeFunc!=null&&r.disposeFunc(this.registry[t])})}initializeBackend(t){const n=this.registryFactory[t];if(n==null)throw new Error(`Cannot initialize backend ${t}, no registration found.`);try{const r=n.factory();if(r&&!(r instanceof Er)&&typeof r.then=="function"){const s=++this.pendingBackendInitId,o=r.then(i=>s<this.pendingBackendInitId?!1:(this.registry[t]=i,this.pendingBackendInit=null,!0)).catch(i=>(s<this.pendingBackendInitId||(this.pendingBackendInit=null,ft(`Initialization of backend ${t} failed`),ft(i.stack||i.message)),!1));return this.pendingBackendInit=o,{success:o,asyncInit:!0}}else return this.registry[t]=r,{success:!0,asyncInit:!1}}catch(r){return ft(`Initialization of backend ${t} failed`),ft(r.stack||r.message),{success:!1,asyncInit:!1}}}removeBackend(t){if(!(t in this.registryFactory))throw new Error(`${t} backend not found in registry`);this.backendName===t&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,t in this.registry&&(this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t]),delete this.registryFactory[t],this.backendName===t&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((t,n)=>this.registryFactory[n].priority-this.registryFactory[t].priority)}initializeBackendsAndReturnBest(){const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const r=t[n],{success:s,asyncInit:o}=this.initializeBackend(r);if(o||s)return{name:r,asyncInit:o}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(t,n){const r=this.state.tensorInfo.get(n),s=r.backend,o=this.readSync(n),i=s.refCount(n);s.disposeData(n,!0),r.backend=t,t.move(n,o,r.shape,r.dtype,i),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(t,n){let r=null;if(n==null){if(typeof t!="function")throw new Error("Please provide a function to tidy()");n=t}else{if(typeof t!="string"&&!(t instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof n!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");r=t}let s;return this.scopedRun(()=>this.startScope(r),()=>this.endScope(s),()=>(s=n(),s instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),s))}scopedRun(t,n,r){t();try{const s=r();return n(),s}catch(s){throw n(),s}}nextTensorId(){return _t.nextTensorId++}nextVariableId(){return _t.nextVariableId++}clone(t){const n=E.runKernel(Mn,{x:t}),r={x:t},s=i=>({x:()=>{const c="float32",a={x:i},l={dtype:c};return E.runKernel($n,a,l)}}),o=[];return this.addTapeNode(this.state.activeScope.name,r,[n],s,o,{}),n}runKernel(t,n,r){if(this.backendName==null&&this.backend,!(en(t,this.backendName)!=null))throw new Error(`Kernel '${t}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:t,inputs:n,attrs:r})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(t,n,r){const s=this.backend.numDataIds();let o=0;r.forEach(a=>{o+=a.dtype==="complex64"?3:1});const i=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],c=s-n-o-i;if(c>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${c} data ids) after running '${t}'`)}runKernelFunc(t){let n,r=[];const s=this.isTapeOn(),o=this.state.numBytes,i=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let c;this.backendName==null&&this.backend;let a;const l=ce(t)?t.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(ce(t)){const{kernelName:g,inputs:p,attrs:m}=t;this.backendName==null&&this.backend;const b=en(g,this.backendName);x(b!=null,()=>`Cannot find registered kernel '${g}' for backend '${this.backendName}'`),c=()=>{const R=this.backend.numDataIds();a=b.kernelFunc({inputs:p,attrs:m,backend:this.backend});const w=Array.isArray(a)?a:[a];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(g,R,w);const S=w.map(I=>I.rank!=null?I:this.makeTensorFromTensorInfo(I));if(s){const I=this.getTensorsForGradient(g,p,S);r=this.saveTensorsForBackwardMode(I)}return S}}else{const{forwardFunc:g}=t,p=m=>{s&&(r=m.map(b=>this.keep(this.clone(b))))};c=()=>{const m=this.backend.numDataIds();a=this.tidy(()=>g(this.backend,p));const b=Array.isArray(a)?a:[a];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(l,m,b),b}}const{inputs:h,attrs:u}=t,f=ce(t)?null:t.backwardsFunc;let d;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?n=c():(d=this.profiler.profileKernel(l,h,()=>c()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(d),n=d.outputs)}),s&&this.addTapeNode(l,h,n,f,r,u),this.state.profiling&&this.state.activeProfile.kernels.push({name:l,bytesAdded:this.state.numBytes-o,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-i,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(h).map(g=>h[g]!=null?h[g].shape:null),outputShapes:n.map(g=>g.shape),kernelTimeMs:d.timeMs,extraInfo:d.extraInfo}),Array.isArray(a)?n:n[0]}saveTensorsForBackwardMode(t){return t.map(r=>this.keep(this.clone(r)))}getTensorsForGradient(t,n,r){const s=nn(t);if(s!=null){const o=s.inputsToSave||[],i=s.outputsToSave||[];let c;s.saveAllInputs?(x(Array.isArray(n),()=>"saveAllInputs is true, expected inputs to be an array."),c=Object.keys(n).map(l=>n[l])):c=o.map(l=>n[l]);const a=r.filter((l,h)=>i[h]);return c.concat(a)}return[]}makeTensor(t,n,r,s){if(t==null)throw new Error("Values passed to engine.makeTensor() are null");r=r||"float32",s=s||this.backend;let o=t;r==="string"&&Ge(t[0])&&(o=t.map(a=>je(a)));const i=s.write(o,n,r),c=new Q(n,r,i,this.nextTensorId());if(this.trackTensor(c,s),r==="string"){const a=this.state.tensorInfo.get(i),l=vr(o);this.state.numBytes+=l-a.bytes,a.bytes=l}return c}makeTensorFromDataId(t,n,r,s){r=r||"float32";const o={dataId:t,shape:n,dtype:r};return this.makeTensorFromTensorInfo(o,s)}makeTensorFromTensorInfo(t,n){const{dataId:r,shape:s,dtype:o}=t,i=new Q(s,o,r,this.nextTensorId());return this.trackTensor(i,n),i}makeVariable(t,n=!0,r,s){r=r||this.nextVariableId().toString(),s!=null&&s!==t.dtype&&(t=t.cast(s));const o=new te(t,n,r,this.nextTensorId());if(this.state.registeredVariables[o.name]!=null)throw new Error(`Variable with name ${o.name} was already registered`);return this.state.registeredVariables[o.name]=o,this.incRef(o,this.backend),o}trackTensor(t,n){this.state.numTensors++,t.dtype==="string"&&this.state.numStringTensors++;let r=0;t.dtype!=="complex64"&&t.dtype!=="string"&&(r=t.size*ge(t.dtype)),this.state.numBytes+=r,this.state.tensorInfo.has(t.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(t.dataId,{backend:n||this.backend,dtype:t.dtype,shape:t.shape,bytes:r})),t instanceof te||this.track(t)}incRef(t,n){this.trackTensor(t,n),this.backend.incRef(t.dataId)}removeDataId(t,n){this.state.tensorInfo.has(t)&&this.state.tensorInfo.get(t).backend===n&&(this.state.tensorInfo.delete(t),this.state.numDataBuffers--)}disposeTensor(t){if(!this.state.tensorInfo.has(t.dataId))return;const n=this.state.tensorInfo.get(t.dataId);if(this.state.numTensors--,t.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=n.bytes),t.dtype!=="complex64"&&t.dtype!=="string"){const r=t.size*ge(t.dtype);this.state.numBytes-=r}n.backend.disposeData(t.dataId)&&this.removeDataId(t.dataId,n.backend)}disposeVariables(){for(const t in this.state.registeredVariables){const n=this.state.registeredVariables[t];this.disposeVariable(n)}}disposeVariable(t){this.disposeTensor(t),this.state.registeredVariables[t.name]!=null&&delete this.state.registeredVariables[t.name]}memory(){const t=this.backend.memory();return t.numTensors=this.state.numTensors,t.numDataBuffers=this.state.numDataBuffers,t.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(t.unreliable=!0,t.reasons==null&&(t.reasons=[]),t.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),t}async profile(t){this.state.profiling=!0;const n=this.state.numBytes,r=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await t(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(s=>s.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-n,this.state.activeProfile.newTensors=this.state.numTensors-r;for(const s of this.state.activeProfile.kernels)s.kernelTimeMs=await s.kernelTimeMs,s.extraInfo=await s.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(t,n,r,s,o,i){const c={id:this.state.nextTapeNodeId++,kernelName:t,inputs:n,outputs:r,saved:o},a=nn(t);a!=null&&(s=a.gradFunc),s!=null&&(c.gradient=l=>(l=l.map((h,u)=>{if(h==null){const f=r[u],d=An(f.size,f.dtype);return this.makeTensor(d,f.shape,f.dtype)}return h}),s(l.length>1?l:l[0],o,i))),this.state.activeTape.push(c)}keep(t){return t.kept=!0,t}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(t){const n={track:[],name:"unnamed scope",id:this.state.nextScopeId++};t&&(n.name=t),this.state.scopeStack.push(n),this.state.activeScope=n}endScope(t){const n=Gn(t),r=new Set(n.map(o=>o.id));for(let o=0;o<this.state.activeScope.track.length;o++){const i=this.state.activeScope.track[o];!i.kept&&!r.has(i.id)&&i.dispose()}const s=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],n.forEach(o=>{!o.kept&&o.scopeId===s.id&&this.track(o)})}gradients(t,n,r,s=!1){if(x(n.length>0,()=>"gradients() received an empty list of xs."),r!=null&&r.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${r.dtype}'`);const o=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",t));x(o instanceof Q,()=>"The result y returned by f() must be a tensor.");const i=ms(this.state.activeTape,n,o);if(!s&&i.length===0&&n.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const c={};c[o.id]=r??vs(o.shape),ys(c,i,l=>this.tidy(l),Ts);const a=n.map(l=>c[l.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(l=>{for(const h of l.saved)h.dispose()}),this.state.activeTape=null),{value:o,grads:a}})}customGrad(t){return x(pe(t),()=>"The f passed in customGrad(f) must be a function."),(...n)=>{x(n.every(c=>c instanceof Q),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let r;const s={};n.forEach((c,a)=>{s[a]=c});const o=(c,a)=>(r=t(...n,a),x(r.value instanceof Q,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),x(pe(r.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),r.value),i=(c,a)=>{const l=r.gradFunc(c,a),h=Array.isArray(l)?l:[l];x(h.length===n.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),x(h.every(f=>f instanceof Q),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const u={};return h.forEach((f,d)=>{u[d]=()=>f}),u};return this.runKernelFunc({forwardFunc:o,backwardsFunc:i,inputs:s})}}readSync(t){return this.state.tensorInfo.get(t).backend.readSync(t)}read(t){return this.state.tensorInfo.get(t).backend.read(t)}readToGPU(t,n){return this.state.tensorInfo.get(t).backend.readToGPU(t,n)}async time(t){const n=Qt(),r=await this.backend.time(t);return r.wallMs=Qt()-n,r}track(t){return this.state.activeScope!=null&&(t.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(t)),t}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new dn;for(const t in this.registry)this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}_t.nextTensorId=0;_t.nextVariableId=0;function vs(e){const t=Mr(V(e),"float32");return E.makeTensor(t,e,"float32")}function zn(){const e=vn();if(e._tfengine==null){const t=new Fr(e);e._tfengine=new _t(t)}return Nr(e._tfengine.ENV),Es(()=>e._tfengine),e._tfengine}const E=zn();function Ts(e,t){const n={a:e,b:t};return E.runKernel(Tn,n)}/**
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
 */function $s(){return typeof navigator<"u"&&navigator!=null}function ff(e){if(e||$s()){if(e||(e=navigator),e.product==="ReactNative")return!0;const t=e.userAgent||e.vendor||(typeof window<"u"?window.opera:"");if(!t){const n=e;return n.userAgentData&&n.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function Ms(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}/**
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
 */const j=v();j.registerFlag("DEBUG",()=>!1,e=>{e&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});j.registerFlag("IS_BROWSER",()=>Ms());j.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");j.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));j.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor));j.registerFlag("PROD",()=>!1);j.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>j.getBool("DEBUG"));j.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);j.registerFlag("IS_TEST",()=>!1);j.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>j.getBool("DEBUG"));j.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);j.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);j.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
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
 */function qn(e,t){let n=e;if(st(e))return t==="string"?[]:[e.length];if(Pn(e)){const s=e.channels||"RGBA";return[e.height,e.width*s.length]}else if(Un(e))return[e.buffer.size/(t==null?4:ge(t))];if(!Array.isArray(e))return[];const r=[];for(;Array.isArray(n)||st(n)&&t!=="string";)r.push(n.length),n=n[0];return Array.isArray(e)&&v().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&Vn(e,r,[]),r}function Vn(e,t,n){if(n=n||[],!Array.isArray(e)&&!st(e)){x(t.length===0,()=>`Element arr[${n.join("][")}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}x(t.length>0,()=>`Element arr[${n.join("][")}] should be a primitive, but is an array of ${e.length} elements`),x(e.length===t[0],()=>`Element arr[${n.join("][")}] should have ${t[0]} elements, but has ${e.length} elements`);const r=t.slice(1);for(let s=0;s<e.length;++s)Vn(e[s],r,n.concat(s))}function gn(e,t,n,r){if(e!=="string_or_numeric"){if(e==null)throw new Error("Expected dtype cannot be null.");if(e!=="numeric"&&e!==t||e==="numeric"&&t==="string")throw new Error(`Argument '${n}' passed to '${r}' must be ${e} tensor, but got ${t} tensor`)}}function k(e,t,n,r="numeric"){if(e instanceof Q)return gn(r,e.dtype,t,n),e;let s=re(e);if(s!=="string"&&["bool","int32","float32"].indexOf(r)>=0&&(s=r),gn(r,s,t,n),e==null||!st(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string"){const a=e==null?"null":e.constructor.name;throw new Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${a}'`)}const o=qn(e,s);!st(e)&&!Array.isArray(e)&&(e=[e]);const c=s!=="string"?Ve(e,s):Wt(e,[],!0);return E.makeTensor(c,o,s)}function hf(e,t,n,r="numeric"){if(!Array.isArray(e))throw new Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((o,i)=>k(o,`${t}[${i}]`,n,r))}/**
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
 */const Fs="__op";function O(e){const t=Object.keys(e);if(t.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0];const r=e[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n=n+Fs;const s=(...o)=>{E.startScope(n);try{const i=r(...o);return We(i)&&console.error("Cannot return a Promise inside of tidy."),E.endScope(i),i}catch(i){throw E.endScope(null),i}};return Object.defineProperty(s,"name",{value:n,configurable:!0}),s}/**
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
 */function Rs(e,t){const n=k(e,"real","complex"),r=k(t,"imag","complex");xr(n.shape,r.shape,`real and imag shapes, ${n.shape} and ${r.shape}, must match in call to tf.complex().`);const s={real:n,imag:r};return E.runKernel(Cr,s)}const Bs=O({complex_:Rs});/**
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
 */function jn(e,t,n,r){if(r==null)r=re(e);else if(r==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(Un(e)||Pn(e)){if(r!=="float32"&&r!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${r}.`);return E.backend.createTensorFromGPUData(e,t||n,r)}if(!st(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(t!=null){se(t);const s=V(t),o=V(n);x(s===o,()=>`Based on the provided shape, [${t}], the tensor should have ${s} values but has ${o}`);for(let i=0;i<n.length;++i){const c=n[i],a=i===n.length-1?c!==V(t.slice(i)):!0;x(n[i]===t[i]||!a,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!st(e)&&!Array.isArray(e)&&(e=[e]),t=t||n,e=r!=="string"?Ve(e,r):Wt(e,[],!0),E.makeTensor(e,t,r)}/**
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
 */function le(e,t,n){const r=qn(e,n);return jn(e,t,r,n)}/**
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
 */const pn={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};/**
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
 */const ee=4;async function df(e,t){const n=[],r=[],s=Array.isArray(e)?e.map(i=>i.name):Object.keys(e);for(let i=0;i<s.length;++i){const c=s[i],a=Array.isArray(e)?e[i].tensor:e[c];if(a.dtype!=="float32"&&a.dtype!=="int32"&&a.dtype!=="bool"&&a.dtype!=="string"&&a.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${c}': ${a.dtype}`);const l={name:c,shape:a.shape,dtype:a.dtype};if(a.dtype==="string"){const h=new Promise(async u=>{const f=await a.bytes(),d=f.reduce((m,b)=>m+b.length,0)+ee*f.length,g=new Uint8Array(d);let p=0;for(let m=0;m<f.length;m++){const b=f[m],R=new Uint8Array(new Uint32Array([b.length]).buffer);g.set(R,p),p+=ee,g.set(b,p),p+=b.length}u(g)});r.push(h)}else r.push(a.data());t!=null&&(l.group=t),n.push(l)}const o=await Promise.all(r);return{data:Ds(o),specs:n}}function gf(e,t){const n={};let r,s=0;for(const o of t){const i=o.name,c=o.dtype,a=o.shape,l=V(a);let h;if("quantization"in o){const u=o.quantization;if(u.dtype==="uint8"||u.dtype==="uint16"){if(!("min"in u&&"scale"in u))throw new Error(`Weight ${o.name} with quantization ${u.dtype} doesn't have corresponding metadata min and scale.`)}else if(u.dtype==="float16"){if(c!=="float32")throw new Error(`Weight ${o.name} is quantized with ${u.dtype} which only supports weights of type float32 not ${c}.`)}else throw new Error(`Weight ${o.name} has unknown quantization dtype ${u.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);const f=pn[u.dtype],d=e.slice(s,s+l*f),g=u.dtype==="uint8"?new Uint8Array(d):new Uint16Array(d);if(c==="float32")if(u.dtype==="uint8"||u.dtype==="uint16"){h=new Float32Array(g.length);for(let p=0;p<g.length;p++){const m=g[p];h[p]=m*u.scale+u.min}}else if(u.dtype==="float16")r===void 0&&(r=Us()),h=r(g);else throw new Error(`Unsupported quantization type ${u.dtype} for weight type float32.`);else if(c==="int32"){if(u.dtype!=="uint8"&&u.dtype!=="uint16")throw new Error(`Unsupported quantization type ${u.dtype} for weight type int32.`);h=new Int32Array(g.length);for(let p=0;p<g.length;p++){const m=g[p];h[p]=Math.round(m*u.scale+u.min)}}else throw new Error(`Unsupported dtype in weight '${i}': ${c}`);s+=l*f}else if(c==="string"){const u=V(o.shape);h=[];for(let f=0;f<u;f++){const d=new Uint32Array(e.slice(s,s+ee))[0];s+=ee;const g=new Uint8Array(e.slice(s,s+d));h.push(g),s+=d}}else{const u=pn[c],f=e.slice(s,s+l*u);if(c==="float32")h=new Float32Array(f);else if(c==="int32")h=new Int32Array(f);else if(c==="bool")h=new Uint8Array(f);else if(c==="complex64"){h=new Float32Array(f);const d=new Float32Array(h.length/2),g=new Float32Array(h.length/2);for(let b=0;b<d.length;b++)d[b]=h[b*2],g[b]=h[b*2+1];const p=le(d,a,"float32"),m=le(g,a,"float32");n[i]=Bs(p,m),p.dispose(),m.dispose()}else throw new Error(`Unsupported dtype in weight '${i}': ${c}`);s+=l*u}c!=="complex64"&&(n[i]=le(h,a,c))}return n}function Ds(e){if(e===null)throw new Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0;const n=[];e.forEach(o=>{if(t+=o.byteLength,n.push(o.byteLength===o.buffer.byteLength?o:new o.constructor(o)),!(o instanceof Float32Array||o instanceof Int32Array||o instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${o.constructor.name}`)});const r=new Uint8Array(t);let s=0;return n.forEach(o=>{r.set(new Uint8Array(o.buffer),s),s+=o.byteLength}),r.buffer}const Ke=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function mn(e){return Ke?Buffer.byteLength(e):new Blob([e]).size}function Ns(e){if(Ke)return Buffer.from(e).toString("base64");const t=new Uint8Array(e);let n="";for(let r=0,s=t.length;r<s;r++)n+=String.fromCharCode(t[r]);return btoa(n)}function _s(e){if(Ke){const r=Buffer.from(e,"base64");return r.buffer.slice(r.byteOffset,r.byteOffset+r.byteLength)}const t=atob(e),n=new Uint8Array(t.length);for(let r=0;r<t.length;++r)n.set([t.charCodeAt(r)],r);return n.buffer}function pf(e){if(e.length===1)return e[0];let t=0;e.forEach(s=>{t+=s.byteLength});const n=new Uint8Array(t);let r=0;return e.forEach(s=>{n.set(new Uint8Array(s),r),r+=s.byteLength}),n.buffer}function mf(e){const t="/";for(e=e.trim();e.endsWith(t);)e=e.slice(0,e.length-1);const n=e.split(t);return n[n.length-1]}function yf(e,t){const n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return e.signature!=null&&(n.signature=e.signature),e.userDefinedMetadata!=null&&(n.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(n.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(n.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(n.trainingConfig=e.trainingConfig),n}function Os(e,t,n){const r={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(e.trainingConfig!=null&&(r.trainingConfig=e.trainingConfig),e.weightsManifest!=null){if(!t)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!n)throw new Error("modelJSON has weightsManifest but weightData is null");r.weightSpecs=t,r.weightData=n}return e.signature!=null&&(r.signature=e.signature),e.userDefinedMetadata!=null&&(r.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(r.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(r.initializerSignature=e.initializerSignature),r}async function wf(e,t){let n,r;return e.weightsManifest!=null&&([n,r]=await t(e.weightsManifest)),Os(e,n,r)}function Hn(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:e.modelTopology==null?0:mn(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:mn(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:e.weightData.byteLength}}function bf(e){const t=[];for(const n of e)t.push(...n.weights);return t}function Cs(){const e=n=>{let r=n<<13,s=0;for(;!(r&8388608);)s-=8388608,r<<=1;return r&=-8388609,s+=947912704,r|s},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let n=1024;n<2048;n++)t[n]=939524096+(n-1024<<13);return t}function Ls(){const e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}function Ps(){const e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}function Us(){const e=Cs(),t=Ls(),n=Ps();return r=>{const s=new ArrayBuffer(4*r.length),o=new Uint32Array(s);for(let i=0;i<r.length;i++){const c=r[i],a=e[n[c>>10]+(c&1023)]+t[c>>10];o[i]=a}return new Float32Array(s)}}/**
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
 */class L{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return L.instance==null&&(L.instance=new L),L.instance}static registerSaveRouter(t){L.getInstance().saveRouters.push(t)}static registerLoadRouter(t){L.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return L.getHandlers(t,"save")}static getLoadHandlers(t,n){return L.getHandlers(t,"load",n)}static getHandlers(t,n,r){const s=[];return(n==="load"?L.getInstance().loadRouters:L.getInstance().saveRouters).forEach(i=>{const c=i(t,r);c!==null&&s.push(c)}),s}}const Sf=e=>L.registerSaveRouter(e),Ef=e=>L.registerLoadRouter(e),xf=e=>L.getSaveHandlers(e),If=(e,t)=>L.getLoadHandlers(e,t);/**
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
 */const ve="tensorflowjs",Te=1,wt="models_store",ht="model_info_store";function Kn(){if(!v().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const e=typeof window>"u"?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw new Error("The current browser does not appear to support IndexedDB.");return t}function $e(e){const t=e.result;t.createObjectStore(wt,{keyPath:"modelPath"}),t.createObjectStore(ht,{keyPath:"modelPath"})}class St{constructor(t){if(this.indexedDB=Kn(),t==null||!t)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=t}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,t)}async load(){return this.databaseAction(this.modelPath)}databaseAction(t,n){return new Promise((r,s)=>{const o=this.indexedDB.open(ve,Te);o.onupgradeneeded=()=>$e(o),o.onsuccess=()=>{const i=o.result;if(n==null){const c=i.transaction(wt,"readonly"),l=c.objectStore(wt).get(this.modelPath);l.onsuccess=()=>{if(l.result==null)return i.close(),s(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));r(l.result.modelArtifacts)},l.onerror=h=>(i.close(),s(l.error)),c.oncomplete=()=>i.close()}else{const c=Hn(n),a=i.transaction(ht,"readwrite");let l=a.objectStore(ht),h;try{h=l.put({modelPath:this.modelPath,modelArtifactsInfo:c})}catch(f){return s(f)}let u;h.onsuccess=()=>{u=i.transaction(wt,"readwrite");const f=u.objectStore(wt);let d;try{d=f.put({modelPath:this.modelPath,modelArtifacts:n,modelArtifactsInfo:c})}catch(g){return s(g)}d.onsuccess=()=>r({modelArtifactsInfo:c}),d.onerror=g=>{l=a.objectStore(ht);const p=l.delete(this.modelPath);p.onsuccess=()=>(i.close(),s(d.error)),p.onerror=m=>(i.close(),s(d.error))}},h.onerror=f=>(i.close(),s(h.error)),a.oncomplete=()=>{u==null?i.close():u.oncomplete=()=>i.close()}}},o.onerror=i=>s(o.error)})}}St.URL_SCHEME="indexeddb://";const Xn=e=>v().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(St.URL_SCHEME)?Gs(e.slice(St.URL_SCHEME.length)):null;L.registerSaveRouter(Xn);L.registerLoadRouter(Xn);function Gs(e){return new St(e)}function Ws(e){return e.startsWith(St.URL_SCHEME)?e.slice(St.URL_SCHEME.length):e}class zs{constructor(){this.indexedDB=Kn()}async listModels(){return new Promise((t,n)=>{const r=this.indexedDB.open(ve,Te);r.onupgradeneeded=()=>$e(r),r.onsuccess=()=>{const s=r.result,o=s.transaction(ht,"readonly"),c=o.objectStore(ht).getAll();c.onsuccess=()=>{const a={};for(const l of c.result)a[l.modelPath]=l.modelArtifactsInfo;t(a)},c.onerror=a=>(s.close(),n(c.error)),o.oncomplete=()=>s.close()},r.onerror=s=>n(r.error)})}async removeModel(t){return t=Ws(t),new Promise((n,r)=>{const s=this.indexedDB.open(ve,Te);s.onupgradeneeded=()=>$e(s),s.onsuccess=()=>{const o=s.result,i=o.transaction(ht,"readwrite"),c=i.objectStore(ht),a=c.get(t);let l;a.onsuccess=()=>{if(a.result==null)return o.close(),r(new Error(`Cannot find model with path '${t}' in IndexedDB.`));{const h=c.delete(t),u=()=>{l=o.transaction(wt,"readwrite");const d=l.objectStore(wt).delete(t);d.onsuccess=()=>n(a.result.modelArtifactsInfo),d.onerror=g=>r(a.error)};h.onsuccess=u,h.onerror=f=>(u(),o.close(),r(a.error))}},a.onerror=h=>(o.close(),r(a.error)),i.oncomplete=()=>{l==null?o.close():l.oncomplete=()=>o.close()}},s.onerror=o=>r(s.error)})}}/**
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
 */const lt="/",Mt="tensorflowjs_models",Zn="info",qs="model_topology",Vs="weight_specs",js="weight_data",Hs="model_metadata";function Jn(e){return{info:[Mt,e,Zn].join(lt),topology:[Mt,e,qs].join(lt),weightSpecs:[Mt,e,Vs].join(lt),weightData:[Mt,e,js].join(lt),modelMetadata:[Mt,e,Hs].join(lt)}}function Yn(e){for(const t of Object.values(e))window.localStorage.removeItem(t)}function Ks(e){const t=e.split(lt);if(t.length<3)throw new Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(lt)}function Xs(e){return e.startsWith(Et.URL_SCHEME)?e.slice(Et.URL_SCHEME.length):e}class Et{constructor(t){if(!v().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,t==null||!t)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=t,this.keys=Jn(this.modelPath)}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const n=JSON.stringify(t.modelTopology),r=JSON.stringify(t.weightSpecs),s=Hn(t);try{this.LS.setItem(this.keys.info,JSON.stringify(s)),this.LS.setItem(this.keys.topology,n),this.LS.setItem(this.keys.weightSpecs,r),this.LS.setItem(this.keys.weightData,Ns(t.weightData));const o={format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,signature:t.signature!=null?t.signature:void 0,userDefinedMetadata:t.userDefinedMetadata!=null?t.userDefinedMetadata:void 0,modelInitializer:t.modelInitializer!=null?t.modelInitializer:void 0,initializerSignature:t.initializerSignature!=null?t.initializerSignature:void 0,trainingConfig:t.trainingConfig!=null?t.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(o)),{modelArtifactsInfo:s}}catch{throw Yn(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${s.modelTopologyBytes}, weightSpecsBytes=${s.weightSpecsBytes}, weightDataBytes=${s.weightDataBytes}.`)}}}async load(){const t=JSON.parse(this.LS.getItem(this.keys.info));if(t==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(t.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const n={},r=JSON.parse(this.LS.getItem(this.keys.topology));if(r==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);n.modelTopology=r;const s=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(s==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);n.weightSpecs=s;const o=this.LS.getItem(this.keys.modelMetadata);if(o!=null){const c=JSON.parse(o);n.format=c.format,n.generatedBy=c.generatedBy,n.convertedBy=c.convertedBy,c.signature!=null&&(n.signature=c.signature),c.userDefinedMetadata!=null&&(n.userDefinedMetadata=c.userDefinedMetadata),c.modelInitializer!=null&&(n.modelInitializer=c.modelInitializer),c.initializerSignature!=null&&(n.initializerSignature=c.initializerSignature),c.trainingConfig!=null&&(n.trainingConfig=c.trainingConfig)}const i=this.LS.getItem(this.keys.weightData);if(i==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return n.weightData=_s(i),n}}Et.URL_SCHEME="localstorage://";const Qn=e=>v().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Et.URL_SCHEME)?Zs(e.slice(Et.URL_SCHEME.length)):null;L.registerSaveRouter(Qn);L.registerLoadRouter(Qn);function Zs(e){return new Et(e)}class Js{constructor(){x(v().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),x(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const t={},n=Mt+lt,r=lt+Zn;for(let s=0;s<this.LS.length;++s){const o=this.LS.key(s);if(o.startsWith(n)&&o.endsWith(r)){const i=Ks(o);t[i]=JSON.parse(this.LS.getItem(o))}}return t}async removeModel(t){t=Xs(t);const n=Jn(t);if(this.LS.getItem(n.info)==null)throw new Error(`Cannot find model at path '${t}'`);const r=JSON.parse(this.LS.getItem(n.info));return Yn(n),r}}/**
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
 */const Rt="://";class z{constructor(){this.managers={}}static getInstance(){return z.instance==null&&(z.instance=new z),z.instance}static registerManager(t,n){x(t!=null,()=>"scheme must not be undefined or null."),t.endsWith(Rt)&&(t=t.slice(0,t.indexOf(Rt))),x(t.length>0,()=>"scheme must not be an empty string.");const r=z.getInstance();x(r.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),r.managers[t]=n}static getManager(t){const n=z.getInstance().managers[t];if(n==null)throw new Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(z.getInstance().managers)}}function Xt(e){if(e.indexOf(Rt)===-1)throw new Error(`The url string provided does not contain a scheme. Supported schemes are: ${z.getSchemes().join(",")}`);return{scheme:e.split(Rt)[0],path:e.split(Rt)[1]}}async function tr(e,t,n=!1){x(e!==t,()=>`Old path and new path are the same: '${e}'`);const r=L.getLoadHandlers(e);x(r.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),x(r.length<2,()=>`Copying failed because more than one (${r.length}) load handlers for source URL ${e}.`);const s=r[0],o=L.getSaveHandlers(t);x(o.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),x(o.length<2,()=>`Copying failed because more than one (${r.length}) save handlers for destination URL ${t}.`);const i=o[0],c=Xt(e).scheme,a=Xt(e).path,l=c===Xt(e).scheme,h=await s.load();n&&l&&await z.getManager(c).removeModel(a);const u=await i.save(h);return n&&!l&&await z.getManager(c).removeModel(a),u.modelArtifactsInfo}async function Af(){const e=z.getSchemes(),t={};for(const n of e){const r=await z.getManager(n).listModels();for(const s in r){const o=n+Rt+s;t[o]=r[s]}}return t}async function kf(e){const t=Xt(e);return z.getManager(t.scheme).removeModel(t.path)}async function vf(e,t){return tr(e,t,!1)}async function Tf(e,t){return tr(e,t,!0)}/**
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
 */class Ys{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(t,n){return fetch(t,n)}now(){return performance.now()}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${n}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(t)}decode(t,n){return new TextDecoder(n).decode(t)}setTimeoutCustom(t,n){if(typeof window>"u"||!v().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(t,n);return}this.functionRefs.push(t),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},n),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",r=>{if(r.source===window&&r.data.name===this.messageName){r.stopPropagation();const s=this.functionRefs[r.data.index];s(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(t){return Rn(t)}}if(v().get("IS_BROWSER")){v().setPlatform("browser",new Ys);try{z.registerManager(Et.URL_SCHEME,new Js)}catch{}try{z.registerManager(St.URL_SCHEME,new zs)}catch{}}/**
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
 */const Qs={importFetch:()=>require("node-fetch")};let ue;class to{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(t,n){return v().global.fetch!=null?v().global.fetch(t,n):(ue==null&&(ue=Qs.importFetch()),ue(t,n))}now(){const t=process.hrtime();return t[0]*1e3+t[1]/1e6}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${n}`);return this.textEncoder.encode(t)}decode(t,n){return t.length===0?"":new this.util.TextDecoder(n).decode(t)}isTypedArray(t){return this.util.types.isFloat32Array(t)||this.util.types.isInt32Array(t)||this.util.types.isUint8Array(t)||this.util.types.isUint8ClampedArray(t)}}v().get("IS_NODE")&&!v().get("IS_BROWSER")&&v().setPlatform("node",new to);/**
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
 */function Me(e,t="float32",n){return t=t||"float32",se(e),new Ss(e,t,n)}/**
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
 */function eo(e,t){const n=k(e,"x","cast");if(!kr(t))throw new Error(`Failed to cast to unknown dtype ${t}`);if(t==="string"&&n.dtype!=="string"||t!=="string"&&n.dtype==="string")throw new Error("Only strings can be casted to strings");const r={x:n},s={dtype:t};return E.runKernel($n,r,s)}const ne=O({cast_:eo});/**
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
 */function no(e){const n={x:k(e,"x","clone","string_or_numeric")};return E.runKernel(Mn,n)}const er=O({clone_:no});/**
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
 */function ro(e,t=!1){console.log(e.toString(t))}/**
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
 */zn();const so={buffer:Me,cast:ne,clone:er,print:ro};xs(so);/**
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
 */function $f(){return E}function Mf(){return E.memory()}function U(e,t){return E.tidy(e,t)}function J(e){Gn(e).forEach(n=>n.dispose())}function oo(e){return E.keep(e)}function Ff(e){return E.setBackend(e)}function Rf(e,t,n=1){return E.registerBackend(e,t,n)}function Bf(){return E.backend}/**
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
 */function io(e,t){let n=k(e,"a","add"),r=k(t,"b","add");[n,r]=kt(n,r);const s={a:n,b:r};return E.runKernel(Tn,s)}const F=O({add_:io});/**
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
 */function ao(e,t){let n=k(e,"a","floorDiv"),r=k(t,"b","floorDiv");[n,r]=kt(n,r);const s={a:n,b:r};return E.runKernel(Wr,s)}const co=O({floorDiv_:ao});/**
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
 */function lo(e,t){let n=k(e,"a","div"),r=k(t,"b","div");if([n,r]=kt(n,r),n.dtype==="int32"&&r.dtype==="int32")return co(n,r);const s={a:n,b:r},o={};return E.runKernel(Pr,s,o)}const at=O({div_:lo});/**
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
 */function uo(e,t){let n=k(e,"a","mul"),r=k(t,"b","mul");[n,r]=kt(n,r);const s={a:n,b:r};return E.runKernel(Vr,s)}const A=O({mul_:uo});/**
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
 */function fo(e){const t=k(e,"x","abs");if(t.dtype==="complex64"){const n={x:t};return E.runKernel(Lr,n)}else{const n={x:t};return E.runKernel(Or,n)}}const ho=O({abs_:fo});/**
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
 */function go(e,t,n,r,s="NHWC",o){const i=e[3],c=[...t,i],a=rr(s);return Xe(e,c,n,o,r,null,null,a)}function po(e,t,n,r,s,o,i="channelsLast"){const[c,a]=zt(t);let l;if(i==="channelsLast")l=[c,a,e[3],e[3]];else if(i==="channelsFirst")l=[c,a,e[1],e[1]];else throw new Error(`Unknown dataFormat ${i}`);return Xe(e,l,n,r,s,o,!1,i)}function mo(e,t,n,r,s,o,i="NDHWC"){const[c,a,l]=Fe(t);let h,u;if(i==="NDHWC")u="channelsLast",h=[c,a,l,e[4],e[4]];else if(i==="NCDHW")u="channelsFirst",h=[c,a,l,e[1],e[1]];else throw new Error(`Unknown dataFormat ${i}`);return nr(e,h,n,r,s,!1,u,o)}function Xe(e,t,n,r,s,o,i=!1,c="channelsLast"){let[a,l,h,u]=[-1,-1,-1,-1];if(c==="channelsLast")[a,l,h,u]=e;else if(c==="channelsFirst")[a,u,l,h]=e;else throw new Error(`Unknown dataFormat ${c}`);const[f,d,,g]=t,[p,m]=zt(n),[b,R]=zt(r),w=Bt(f,b),S=Bt(d,R),{padInfo:I,outHeight:T,outWidth:$}=bo(s,l,h,p,m,w,S,o,c),M=i?g*u:g;let B;return c==="channelsFirst"?B=[a,M,T,$]:c==="channelsLast"&&(B=[a,T,$,M]),{batchSize:a,dataFormat:c,inHeight:l,inWidth:h,inChannels:u,outHeight:T,outWidth:$,outChannels:M,padInfo:I,strideHeight:p,strideWidth:m,filterHeight:f,filterWidth:d,effectiveFilterHeight:w,effectiveFilterWidth:S,dilationHeight:b,dilationWidth:R,inShape:e,outShape:B,filterShape:t}}function nr(e,t,n,r,s,o=!1,i="channelsLast",c){let[a,l,h,u,f]=[-1,-1,-1,-1,-1];if(i==="channelsLast")[a,l,h,u,f]=e;else if(i==="channelsFirst")[a,f,l,h,u]=e;else throw new Error(`Unknown dataFormat ${i}`);const[d,g,p,,m]=t,[b,R,w]=Fe(n),[S,I,T]=Fe(r),$=Bt(d,S),M=Bt(g,I),B=Bt(p,T),{padInfo:G,outDepth:C,outHeight:W,outWidth:H}=So(s,l,h,u,b,R,w,$,M,B,c),Y=o?m*f:m;let ut;return i==="channelsFirst"?ut=[a,Y,C,W,H]:i==="channelsLast"&&(ut=[a,C,W,H,Y]),{batchSize:a,dataFormat:i,inDepth:l,inHeight:h,inWidth:u,inChannels:f,outDepth:C,outHeight:W,outWidth:H,outChannels:Y,padInfo:G,strideDepth:b,strideHeight:R,strideWidth:w,filterDepth:d,filterHeight:g,filterWidth:p,effectiveFilterDepth:$,effectiveFilterHeight:M,effectiveFilterWidth:B,dilationDepth:S,dilationHeight:I,dilationWidth:T,inShape:e,outShape:ut,filterShape:t}}function yo(e,t,n,r,s){r==null&&(r=Ze(e,t,n));const o=e[0],i=e[1],c=qt((o-t+2*r)/n+1,s),a=qt((i-t+2*r)/n+1,s);return[c,a]}function wo(e,t,n,r,s,o){s==null&&(s=Ze(e,t[0],r[0]));const i=[0,0,0,n];for(let c=0;c<3;c++)e[c]+2*s>=t[c]&&(i[c]=qt((e[c]-t[c]+2*s)/r[c]+1,o));return i}function Ze(e,t,n,r=1){const s=Bt(t,r);return Math.floor((e[0]*(n-1)-n+s)/2)}function zt(e){return typeof e=="number"?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function Fe(e){return typeof e=="number"?[e,e,e]:e}function Bt(e,t){return t<=1?e:e+(e-1)*(t-1)}function bo(e,t,n,r,s,o,i,c,a){let l,h,u;if(typeof e=="number"){l={top:e,bottom:e,left:e,right:e,type:e===0?"VALID":"NUMBER"};const d=yo([t,n],o,r,e,c);h=d[0],u=d[1]}else if(e==="same"){h=Math.ceil(t/r),u=Math.ceil(n/s);const f=Math.max(0,(h-1)*r+o-t),d=Math.max(0,(u-1)*s+i-n),g=Math.floor(f/2),p=f-g,m=Math.floor(d/2),b=d-m;l={top:g,bottom:p,left:m,right:b,type:"SAME"}}else if(e==="valid")l={top:0,bottom:0,left:0,right:0,type:"VALID"},h=Math.ceil((t-o+1)/r),u=Math.ceil((n-i+1)/s);else if(typeof e=="object"){const f=a==="channelsLast"?e[1][0]:e[2][0],d=a==="channelsLast"?e[1][1]:e[2][1],g=a==="channelsLast"?e[2][0]:e[3][0],p=a==="channelsLast"?e[2][1]:e[3][1];l={top:f,bottom:d,left:g,right:p,type:f===0&&d===0&&g===0&&p===0?"VALID":"EXPLICIT"},h=qt((t-o+f+d)/r+1,c),u=qt((n-i+g+p)/s+1,c)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:l,outHeight:h,outWidth:u}}function So(e,t,n,r,s,o,i,c,a,l,h){let u,f,d,g;if(e==="valid"&&(e=0),typeof e=="number"){u={top:e,bottom:e,left:e,right:e,front:e,back:e,type:e===0?"VALID":"NUMBER"};const m=wo([t,n,r,1],[c,a,l],1,[s,o,i],e,h);f=m[0],d=m[1],g=m[2]}else if(e==="same"){f=Math.ceil(t/s),d=Math.ceil(n/o),g=Math.ceil(r/i);const p=(f-1)*s+c-t,m=(d-1)*o+a-n,b=(g-1)*i+l-r,R=Math.floor(p/2),w=p-R,S=Math.floor(m/2),I=m-S,T=Math.floor(b/2),$=b-T;u={top:S,bottom:I,left:T,right:$,front:R,back:w,type:"SAME"}}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:u,outDepth:f,outHeight:d,outWidth:g}}function qt(e,t){if(!t)return Math.trunc(e);switch(t){case"round":return Math.round(e);case"ceil":return Math.ceil(e);case"floor":return Math.floor(e);default:throw new Error(`Unknown roundingMode ${t}`)}}function Re(e){const[t,n,r]=zt(e);return t===1&&n===1&&r===1}function Eo(e,t){return Re(e)||Re(t)}function xo(e){return zt(e).every(t=>t>0)}function rr(e){if(e==="NHWC")return"channelsLast";if(e==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${e}`)}function Io(e,t,n){if(n!=null){if(typeof t=="string")throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t=="number")x(de(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t=="object")t.forEach(r=>{r.forEach(s=>{x(de(s),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${s}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}/**
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
 */function Ao(e,t){const r={x:k(e,"x","reshape","string_or_numeric")},s={shape:t};return E.runKernel(Xr,r,s)}const sr=O({reshape_:Ao});/**
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
 */function ko(e){const n={x:k(e,"x","sigmoid","float32")};return E.runKernel(Jr,n)}const vo=O({sigmoid_:ko});/**
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
 */function To(e,t){let n=k(e,"broadcastTo","x");const r=n.shape;if(se(t),t.length<n.rank)throw new Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){const l=n.shape.slice();for(;l.length<t.length;)l.unshift(1);n=sr(n,l)}const s=n.shape,o=Array.from(t);for(let l=t.length-1;l>=0;l--)if(s[l]===t[l])o[l]=1;else if(n.shape[l]!==1)throw new Error(`broadcastTo(): [${r}] cannot be broadcast to [${t}].`);if(o.map((l,h)=>l>1?h:-1).filter(l=>l>=0).length===0)return er(n);const c={x:n},a={reps:o};return E.runKernel(es,c,a)}const Df=O({broadcastTo_:To});/**
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
 */function $o(e,t,n){se(e),n=n||re(t);const r={shape:e,value:t,dtype:n};return E.runKernel(Gr,{},r)}/**
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
 */function Mo(e,t){const n=e.length,r=[];for(let s=0;s<n;s++){const o=n-1-s,i=e[o]||1;(t[t.length-1-s]||1)>1&&i===1&&r.unshift(o)}return r}function or(e,t){const n=[];for(let r=0;r<t.length;r++){const s=e[e.length-r-1],o=t.length-r-1,i=t[o];(s==null||s===1&&i>1)&&n.unshift(o)}return n}function ir(e,t){const n=Math.max(e.length,t.length),r=new Array(n);for(let s=0;s<n;s++){let o=e[e.length-s-1];o==null&&(o=1);let i=t[t.length-s-1];if(i==null&&(i=1),o===1)r[n-s-1]=i;else if(i===1)r[n-s-1]=o;else if(o!==i){const c=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(c)}else r[n-s-1]=o}return r}/**
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
 */function Fo(e){const n={x:k(e,"x","zerosLike")};return E.runKernel(ns,n)}const ct=O({zerosLike_:Fo});/**
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
 */function Ro(e){const n={x:k(e,"x","elu","float32")};return E.runKernel(Ur,n)}const Bo=O({elu_:Ro});/**
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
 */function Je(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function ar(e,t,n){const r=e.length+t.length,s=[];let o=0,i=0;for(let c=0;c<r;c++)n.indexOf(c)===-1?s.push(e[o++]):s.push(t[i++]);return s}function Do(e,t){const n=[],r=e.length;for(let o=0;o<r;o++)t.indexOf(o)===-1&&n.push(e[o]);const s=t.map(o=>e[o]);return[n,s]}function No(e,t){const n=t.map(r=>1);return ar(e,n,t)}function _o(e,t,n){x(Je(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function Oo(e,t){if(Je(e,t))return null;const n=[];for(let r=0;r<t;++r)e.indexOf(r)===-1&&n.push(r);return e.forEach(r=>n.push(r)),n}function Co(e){return e.map((t,n)=>[n,t]).sort((t,n)=>t[1]-n[1]).map(t=>t[0])}function Lo(e,t){const n=[];for(let r=t-e;r<t;++r)n.push(r);return n}/**
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
 */function Po(e,t){let n=k(e,"base","pow"),r=k(t,"exp","pow");[n,r]=kt(n,r);const s={a:n,b:r};return E.runKernel(jr,s)}const yn=O({pow_:Po});/**
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
 */function xt(e,t){if((st(e)&&t!=="string"||Array.isArray(e))&&t!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(t==="string"&&st(e)&&!(e instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return jn(e,[],[],t)}/**
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
 */function Uo(e){const n={x:k(e,"x","sqrt","float32")};return E.runKernel(Yr,n)}const Ot=O({sqrt_:Uo});/**
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
 */function Go(e){const t=k(e,"x","square"),n={};return E.runKernel("Square",{x:t},n)}const bt=O({square_:Go});/**
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
 */function Wo(e,t=null,n=!1){let r=k(e,"x","sum");r.dtype==="bool"&&(r=ne(r,"int32"));const s={x:r},o={axis:t,keepDims:n};return E.runKernel(Qr,s,o)}const zo=O({sum_:Wo});/**
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
 */function qo(e,t=.2){const r={x:k(e,"x","leakyRelu")},s={alpha:t};return E.runKernel(zr,r,s)}const Vo=O({leakyRelu_:qo});/**
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
 */function jo(e,t){x(pe(e),()=>"The f passed in variableGrads(f) must be a function"),x(t==null||Array.isArray(t)&&t.every(l=>l instanceof te),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const n=t!=null;if(!n){t=[];for(const l in E.registeredVariables)t.push(E.registeredVariables[l])}const r=n?t.filter(l=>!l.trainable):null,s=t.length;t=t.filter(l=>l.trainable),x(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${s} variables is trainable.`);const o=!0,{value:i,grads:c}=E.gradients(e,t,null,o);x(c.some(l=>l!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),x(i.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${i.rank} tensor`);const a={};return t.forEach((l,h)=>{c[h]!=null&&(a[l.name]=c[h])}),r!=null&&r.forEach(l=>a[l.name]=null),{value:i,grads:a}}function Nf(e){return E.customGrad(e)}/**
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
 */function Ho(e,t){let n=k(e,"a","sub"),r=k(t,"b","sub");[n,r]=kt(n,r);const s={a:n,b:r};return E.runKernel(ts,s)}const Dt=O({sub_:Ho});/**
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
 */function Ko(e,t){let n=k(e,"a","maximum"),r=k(t,"b","maximum");[n,r]=kt(n,r),n.dtype==="bool"&&(n=ne(n,"int32"),r=ne(r,"int32")),ir(n.shape,r.shape);const s={a:n,b:r};return E.runKernel(qr,s)}const Xo=O({maximum_:Ko});/**
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
 */function Zo(e,t){const n=k(e,"x","prelu"),r=k(t,"alpha","prelu"),s={x:n,alpha:r};return E.runKernel(Hr,s)}const Jo=O({prelu_:Zo});var Be={},Yo={get exports(){return Be},set exports(e){Be=e}};(function(e){(function(t,n,r){function s(a){var l=this,h=c();l.next=function(){var u=2091639*l.s0+l.c*23283064365386963e-26;return l.s0=l.s1,l.s1=l.s2,l.s2=u-(l.c=u|0)},l.c=1,l.s0=h(" "),l.s1=h(" "),l.s2=h(" "),l.s0-=h(a),l.s0<0&&(l.s0+=1),l.s1-=h(a),l.s1<0&&(l.s1+=1),l.s2-=h(a),l.s2<0&&(l.s2+=1),h=null}function o(a,l){return l.c=a.c,l.s0=a.s0,l.s1=a.s1,l.s2=a.s2,l}function i(a,l){var h=new s(a),u=l&&l.state,f=h.next;return f.int32=function(){return h.next()*4294967296|0},f.double=function(){return f()+(f()*2097152|0)*11102230246251565e-32},f.quick=f,u&&(typeof u=="object"&&o(u,h),f.state=function(){return o(h,{})}),f}function c(){var a=4022871197,l=function(h){h=String(h);for(var u=0;u<h.length;u++){a+=h.charCodeAt(u);var f=.02519603282416938*a;a=f>>>0,f-=a,f*=a,a=f>>>0,f-=a,a+=f*4294967296}return(a>>>0)*23283064365386963e-26};return l}n&&n.exports?n.exports=i:r&&r.amd?r(function(){return i}):this.alea=i})(It,e,!1)})(Yo);var De={},Qo={get exports(){return De},set exports(e){De=e}};(function(e){(function(t,n,r){function s(c){var a=this,l="";a.x=0,a.y=0,a.z=0,a.w=0,a.next=function(){var u=a.x^a.x<<11;return a.x=a.y,a.y=a.z,a.z=a.w,a.w^=a.w>>>19^u^u>>>8},c===(c|0)?a.x=c:l+=c;for(var h=0;h<l.length+64;h++)a.x^=l.charCodeAt(h)|0,a.next()}function o(c,a){return a.x=c.x,a.y=c.y,a.z=c.z,a.w=c.w,a}function i(c,a){var l=new s(c),h=a&&a.state,u=function(){return(l.next()>>>0)/4294967296};return u.double=function(){do var f=l.next()>>>11,d=(l.next()>>>0)/4294967296,g=(f+d)/(1<<21);while(g===0);return g},u.int32=l.next,u.quick=u,h&&(typeof h=="object"&&o(h,l),u.state=function(){return o(l,{})}),u}n&&n.exports?n.exports=i:r&&r.amd?r(function(){return i}):this.xor128=i})(It,e,!1)})(Qo);var Ne={},ti={get exports(){return Ne},set exports(e){Ne=e}};(function(e){(function(t,n,r){function s(c){var a=this,l="";a.next=function(){var u=a.x^a.x>>>2;return a.x=a.y,a.y=a.z,a.z=a.w,a.w=a.v,(a.d=a.d+362437|0)+(a.v=a.v^a.v<<4^(u^u<<1))|0},a.x=0,a.y=0,a.z=0,a.w=0,a.v=0,c===(c|0)?a.x=c:l+=c;for(var h=0;h<l.length+64;h++)a.x^=l.charCodeAt(h)|0,h==l.length&&(a.d=a.x<<10^a.x>>>4),a.next()}function o(c,a){return a.x=c.x,a.y=c.y,a.z=c.z,a.w=c.w,a.v=c.v,a.d=c.d,a}function i(c,a){var l=new s(c),h=a&&a.state,u=function(){return(l.next()>>>0)/4294967296};return u.double=function(){do var f=l.next()>>>11,d=(l.next()>>>0)/4294967296,g=(f+d)/(1<<21);while(g===0);return g},u.int32=l.next,u.quick=u,h&&(typeof h=="object"&&o(h,l),u.state=function(){return o(l,{})}),u}n&&n.exports?n.exports=i:r&&r.amd?r(function(){return i}):this.xorwow=i})(It,e,!1)})(ti);var _e={},ei={get exports(){return _e},set exports(e){_e=e}};(function(e){(function(t,n,r){function s(c){var a=this;a.next=function(){var h=a.x,u=a.i,f,d;return f=h[u],f^=f>>>7,d=f^f<<24,f=h[u+1&7],d^=f^f>>>10,f=h[u+3&7],d^=f^f>>>3,f=h[u+4&7],d^=f^f<<7,f=h[u+7&7],f=f^f<<13,d^=f^f<<9,h[u]=d,a.i=u+1&7,d};function l(h,u){var f,d=[];if(u===(u|0))d[0]=u;else for(u=""+u,f=0;f<u.length;++f)d[f&7]=d[f&7]<<15^u.charCodeAt(f)+d[f+1&7]<<13;for(;d.length<8;)d.push(0);for(f=0;f<8&&d[f]===0;++f);for(f==8?d[7]=-1:d[f],h.x=d,h.i=0,f=256;f>0;--f)h.next()}l(a,c)}function o(c,a){return a.x=c.x.slice(),a.i=c.i,a}function i(c,a){c==null&&(c=+new Date);var l=new s(c),h=a&&a.state,u=function(){return(l.next()>>>0)/4294967296};return u.double=function(){do var f=l.next()>>>11,d=(l.next()>>>0)/4294967296,g=(f+d)/(1<<21);while(g===0);return g},u.int32=l.next,u.quick=u,h&&(h.x&&o(h,l),u.state=function(){return o(l,{})}),u}n&&n.exports?n.exports=i:r&&r.amd?r(function(){return i}):this.xorshift7=i})(It,e,!1)})(ei);var Oe={},ni={get exports(){return Oe},set exports(e){Oe=e}};(function(e){(function(t,n,r){function s(c){var a=this;a.next=function(){var h=a.w,u=a.X,f=a.i,d,g;return a.w=h=h+1640531527|0,g=u[f+34&127],d=u[f=f+1&127],g^=g<<13,d^=d<<17,g^=g>>>15,d^=d>>>12,g=u[f]=g^d,a.i=f,g+(h^h>>>16)|0};function l(h,u){var f,d,g,p,m,b=[],R=128;for(u===(u|0)?(d=u,u=null):(u=u+"\0",d=0,R=Math.max(R,u.length)),g=0,p=-32;p<R;++p)u&&(d^=u.charCodeAt((p+32)%u.length)),p===0&&(m=d),d^=d<<10,d^=d>>>15,d^=d<<4,d^=d>>>13,p>=0&&(m=m+1640531527|0,f=b[p&127]^=d+m,g=f==0?g+1:0);for(g>=128&&(b[(u&&u.length||0)&127]=-1),g=127,p=4*128;p>0;--p)d=b[g+34&127],f=b[g=g+1&127],d^=d<<13,f^=f<<17,d^=d>>>15,f^=f>>>12,b[g]=d^f;h.w=m,h.X=b,h.i=g}l(a,c)}function o(c,a){return a.i=c.i,a.w=c.w,a.X=c.X.slice(),a}function i(c,a){c==null&&(c=+new Date);var l=new s(c),h=a&&a.state,u=function(){return(l.next()>>>0)/4294967296};return u.double=function(){do var f=l.next()>>>11,d=(l.next()>>>0)/4294967296,g=(f+d)/(1<<21);while(g===0);return g},u.int32=l.next,u.quick=u,h&&(h.X&&o(h,l),u.state=function(){return o(l,{})}),u}n&&n.exports?n.exports=i:r&&r.amd?r(function(){return i}):this.xor4096=i})(It,e,!1)})(ni);var Ce={},ri={get exports(){return Ce},set exports(e){Ce=e}};(function(e){(function(t,n,r){function s(c){var a=this,l="";a.next=function(){var u=a.b,f=a.c,d=a.d,g=a.a;return u=u<<25^u>>>7^f,f=f-d|0,d=d<<24^d>>>8^g,g=g-u|0,a.b=u=u<<20^u>>>12^f,a.c=f=f-d|0,a.d=d<<16^f>>>16^g,a.a=g-u|0},a.a=0,a.b=0,a.c=-1640531527,a.d=1367130551,c===Math.floor(c)?(a.a=c/4294967296|0,a.b=c|0):l+=c;for(var h=0;h<l.length+20;h++)a.b^=l.charCodeAt(h)|0,a.next()}function o(c,a){return a.a=c.a,a.b=c.b,a.c=c.c,a.d=c.d,a}function i(c,a){var l=new s(c),h=a&&a.state,u=function(){return(l.next()>>>0)/4294967296};return u.double=function(){do var f=l.next()>>>11,d=(l.next()>>>0)/4294967296,g=(f+d)/(1<<21);while(g===0);return g},u.int32=l.next,u.quick=u,h&&(typeof h=="object"&&o(h,l),u.state=function(){return o(l,{})}),u}n&&n.exports?n.exports=i:r&&r.amd?r(function(){return i}):this.tychei=i})(It,e,!1)})(ri);var Le={},si={get exports(){return Le},set exports(e){Le=e}};const oi={},ii=Object.freeze(Object.defineProperty({__proto__:null,default:oi},Symbol.toStringTag,{value:"Module"})),ai=os(ii);(function(e){(function(t,n,r){var s=256,o=6,i=52,c="random",a=r.pow(s,o),l=r.pow(2,i),h=l*2,u=s-1,f;function d(S,I,T){var $=[];I=I==!0?{entropy:!0}:I||{};var M=b(m(I.entropy?[S,w(n)]:S??R(),3),$),B=new g($),G=function(){for(var C=B.g(o),W=a,H=0;C<l;)C=(C+H)*s,W*=s,H=B.g(1);for(;C>=h;)C/=2,W/=2,H>>>=1;return(C+H)/W};return G.int32=function(){return B.g(4)|0},G.quick=function(){return B.g(4)/4294967296},G.double=G,b(w(B.S),n),(I.pass||T||function(C,W,H,Y){return Y&&(Y.S&&p(Y,B),C.state=function(){return p(B,{})}),H?(r[c]=C,W):C})(G,M,"global"in I?I.global:this==r,I.state)}function g(S){var I,T=S.length,$=this,M=0,B=$.i=$.j=0,G=$.S=[];for(T||(S=[T++]);M<s;)G[M]=M++;for(M=0;M<s;M++)G[M]=G[B=u&B+S[M%T]+(I=G[M])],G[B]=I;($.g=function(C){for(var W,H=0,Y=$.i,ut=$.j,Ct=$.S;C--;)W=Ct[Y=u&Y+1],H=H*s+Ct[u&(Ct[Y]=Ct[ut=u&ut+W])+(Ct[ut]=W)];return $.i=Y,$.j=ut,H})(s)}function p(S,I){return I.i=S.i,I.j=S.j,I.S=S.S.slice(),I}function m(S,I){var T=[],$=typeof S,M;if(I&&$=="object")for(M in S)try{T.push(m(S[M],I-1))}catch{}return T.length?T:$=="string"?S:S+"\0"}function b(S,I){for(var T=S+"",$,M=0;M<T.length;)I[u&M]=u&($^=I[u&M]*19)+T.charCodeAt(M++);return w(I)}function R(){try{var S;return f&&(S=f.randomBytes)?S=S(s):(S=new Uint8Array(s),(t.crypto||t.msCrypto).getRandomValues(S)),w(S)}catch{var I=t.navigator,T=I&&I.plugins;return[+new Date,t,T,t.screen,w(n)]}}function w(S){return String.fromCharCode.apply(0,S)}if(b(r.random(),n),e.exports){e.exports=d;try{f=ai}catch{}}else r["seed"+c]=d})(typeof self<"u"?self:It,[],Math)})(si);var ci=Be,li=De,ui=Ne,fi=_e,hi=Oe,di=Ce,vt=Le;vt.alea=ci;vt.xor128=li;vt.xorwow=ui;vt.xorshift7=fi;vt.xor4096=hi;vt.tychei=di;var _f=vt;/**
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
 */function gi(e){const n={x:k(e,"x","relu")};return E.runKernel(Kr,n)}const pi=O({relu_:gi});/**
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
 */function mi(e){const n={x:k(e,"x","relu6")};return E.runKernel(Zr,n)}const yi=O({relu6_:mi});/**
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
 */function wi(e,t=0){const r={x:k(e,"x","step")},s={alpha:t};return E.runKernel(rs,r,s)}const bi=O({step_:wi});function cr(e,t,n){const r=t.rank>1?t.shape[t.rank-1]:1,s=t.rank>1?t.rank-1:1,o=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${r}, and batchDim: ${s}.`;if(n.rank<s)throw new Error(o+` update.rank < ${s}. `);if(e.length<r+(n.rank-s))throw new Error(o+` Output shape length < ${r+(n.rank-s)}`);if(n.rank!==s+e.length-r)throw new Error(o+` update.rank != ${s+e.length-r}`);for(let i=0;i<s;++i)if(n.shape[i]!==t.shape[i])throw new Error(o+` updates.shape[${i}] (${n.shape[i]}) != indices.shape[${i}] (${t.shape[i]}).`);for(let i=0;i<n.rank-s;++i)if(n.shape[i+s]!==e[i+r])throw new Error(o+` updates.shape[${i+s}] (${n.shape[i+s]}) != shape[${i+s}] (${e[i+s]})`)}function Si(e,t,n){if(t.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.dtype!=="int32")throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(n.length===0){if(t.size===0)throw new Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(e.size===0)throw new Error(`Updates specified for empty output. updates shape: ${e.shape}`)}cr(n,t,e)}function Ei(e,t,n){const r=t.shape.length,s=r>1?t.shape[r-1]:1,o=n.length;let i=1;for(let u=s;u<o;++u)i*=n[u];const c=s<1?1:s,a=V(t.shape)/c,l=[...Vt(n.slice(0,s)),1],h=V(n);return{sliceRank:s,numUpdates:a,sliceSize:i,strides:l,outputSize:h}}/**
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
 */function Of(e,t){const n=[];for(let o=0;o<t.length;o++)t[o]&&n.push(o);const r=Me(e,"int32"),s=Me([n.length,e.length],"int32");for(let o=0;o<n.length;o++){const i=r.indexToLoc(n[o]),c=o*e.length;s.values.set(i,c)}return s.toTensor()}/**
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
 */function xi(e,t,n){if(n==null||n==="linear")return e;if(n==="relu")return A(e,bi(t));throw new Error(`Cannot compute gradient for fused activation ${n}.`)}function Ii(e,t){let n=t;const r=or(e.shape,t.shape);return r.length>0&&(n=zo(n,r)),sr(n,e.shape)}function Ai(e,t,n,r){if(t==="linear")return e;if(t==="relu")return pi(e);if(t==="elu")return Bo(e);if(t==="relu6")return yi(e);if(t==="prelu")return Jo(e,n);if(t==="leakyrelu")return Vo(e,r);if(t==="sigmoid")return vo(e);throw new Error(`Unknown fused activation ${t}.`)}const ki=(e,t)=>!(e>0)||t==="linear";/**
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
 */function vi(e,t,n){const r=Ti(e,t,n),s=r<0?-(r+1):r;e.splice(s,0,t)}function Ti(e,t,n){return Mi(e,t,n||$i)}function $i(e,t){return e>t?1:e<t?-1:0}function Mi(e,t,n){let r=0,s=e.length,o=0,i=!1;for(;r<s;){o=r+(s-r>>>1);const c=n(t,e[o]);c>0?r=o+1:(s=o,i=!c)}return i?r:-r-1}/**
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
 */function Cf(e,t,n,r,s){return Ye(e,t,n,r,s,0)}function Lf(e,t,n,r,s,o){return Ye(e,t,n,r,s,0,!1,o,!0)}function Pf(e,t,n,r,s,o){return Ye(e,t,n,r,s,o,!0)}function Ye(e,t,n,r,s,o,i=!1,c=!1,a=!1){const l=[];for(let m=0;m<t.length;m++)t[m]>s&&l.push({score:t[m],boxIndex:m,suppressBeginIndex:0});l.sort(wn);const h=o>0?-.5/o:0,u=[],f=[];for(;u.length<n&&l.length>0;){const m=l.pop(),{score:b,boxIndex:R,suppressBeginIndex:w}=m;if(b<s)break;let S=!1;for(let I=u.length-1;I>=w;--I){const T=Fi(e,R,u[I]);if(T>=r){S=!0;break}if(m.score=m.score*Ri(r,h,T),m.score<=s)break}m.suppressBeginIndex=u.length,S||(m.score===b?(u.push(R),f.push(m.score)):m.score>s&&vi(l,m,wn))}const d=u.length,g=n-d;c&&g>0&&(u.push(...new Array(g).fill(0)),f.push(...new Array(g).fill(0)));const p={selectedIndices:u};return i&&(p.selectedScores=f),a&&(p.validOutputs=d),p}function Fi(e,t,n){const r=e.subarray(t*4,t*4+4),s=e.subarray(n*4,n*4+4),o=Math.min(r[0],r[2]),i=Math.min(r[1],r[3]),c=Math.max(r[0],r[2]),a=Math.max(r[1],r[3]),l=Math.min(s[0],s[2]),h=Math.min(s[1],s[3]),u=Math.max(s[0],s[2]),f=Math.max(s[1],s[3]),d=(c-o)*(a-i),g=(u-l)*(f-h);if(d<=0||g<=0)return 0;const p=Math.max(o,l),m=Math.max(i,h),b=Math.min(c,u),R=Math.min(a,f),w=Math.max(b-p,0)*Math.max(R-m,0);return w/(d+g-w)}function Ri(e,t,n){const r=Math.exp(t*n*n);return n<=e?r:0}function wn(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}/**
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
 */class Bi{getClassName(){return this.constructor.className}static fromConfig(t,n){return new t(n)}}class mt{constructor(){this.classNameMap={}}static getMap(){return mt.instance==null&&(mt.instance=new mt),mt.instance}static register(t){mt.getMap().classNameMap[t.className]=[t,t.fromConfig]}}function Di(e){x(e.className!=null,()=>"Class being registered does not have the static className property defined."),x(typeof e.className=="string",()=>"className is required to be a string, but got type "+typeof e.className),x(e.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),mt.register(e)}/**
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
 */class Tt extends Bi{minimize(t,n=!1,r){const{value:s,grads:o}=this.computeGradients(t,r);if(r!=null){const i=r.map(c=>({name:c.name,tensor:o[c.name]}));this.applyGradients(i)}else this.applyGradients(o);return J(o),n?s:(s.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(t,n){return jo(t,n)}dispose(){this.iterations_!=null&&J(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:xt(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(t){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(t){return this.iterations_=(await t[0].tensor.data())[0],t.slice(1)}}Object.defineProperty(Tt,Symbol.hasInstance,{value:e=>e.minimize!=null&&e.computeGradients!=null&&e.applyGradients!=null});/**
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
 */class Ni extends Tt{static get className(){return"Adadelta"}constructor(t,n,r=null){super(),this.learningRate=t,this.rho=n,this.epsilon=r,this.accumulatedGrads=[],this.accumulatedUpdates=[],r==null&&(this.epsilon=E.backend.epsilon())}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const o=E.registeredVariables[r],i=!1;this.accumulatedGrads[s]==null&&(this.accumulatedGrads[s]={originalName:`${r}/accum_grad`,variable:U(()=>ct(o).variable(i))}),this.accumulatedUpdates[s]==null&&(this.accumulatedUpdates[s]={originalName:`${r}/accum_var`,variable:U(()=>ct(o).variable(i))});const c=Array.isArray(t)?t[s].tensor:t[r];if(c==null)return;const a=this.accumulatedGrads[s].variable,l=this.accumulatedUpdates[s].variable;U(()=>{const h=F(A(a,this.rho),A(bt(c),1-this.rho)),u=A(at(Ot(F(l,this.epsilon)),Ot(F(a,this.epsilon))),c),f=F(A(l,this.rho),A(bt(u),1-this.rho));a.assign(h),l.assign(f);const d=F(A(u,-this.learningRate),o);o.assign(d)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(J(this.accumulatedGrads.map(t=>t.variable)),J(this.accumulatedUpdates.map(t=>t.variable)))}async getWeights(){const t=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=t.length/2,r=!1;this.accumulatedGrads=t.slice(0,n).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})),this.accumulatedUpdates=t.slice(n,n*2).map(s=>({originalName:s.name,variable:s.tensor.variable(r)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.rho,n.epsilon)}}/**
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
 */class _i extends Tt{static get className(){return"Adagrad"}constructor(t,n=.1){super(),this.learningRate=t,this.initialAccumulatorValue=n,this.accumulatedGrads=[]}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const o=E.registeredVariables[r];this.accumulatedGrads[s]==null&&(this.accumulatedGrads[s]={originalName:`${r}/accumulator`,variable:U(()=>$o(o.shape,this.initialAccumulatorValue).variable(!1))});const i=Array.isArray(t)?t[s].tensor:t[r];if(i==null)return;const c=this.accumulatedGrads[s].variable;U(()=>{const a=F(c,bt(i));c.assign(a);const l=F(A(at(i,Ot(F(a,E.backend.epsilon()))),-this.learningRate),o);o.assign(l)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&J(this.accumulatedGrads.map(t=>t.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulatedGrads=t.map(r=>({originalName:r.name,variable:r.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(t,n){return new t(n.learningRate,n.initialAccumulatorValue)}}/**
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
 */class Oi extends Tt{static get className(){return"Adam"}constructor(t,n,r,s=null){super(),this.learningRate=t,this.beta1=n,this.beta2=r,this.epsilon=s,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],U(()=>{this.accBeta1=xt(n).variable(),this.accBeta2=xt(r).variable()}),s==null&&(this.epsilon=E.backend.epsilon())}applyGradients(t){const n=Array.isArray(t)?t.map(r=>r.name):Object.keys(t);U(()=>{const r=Dt(1,this.accBeta1),s=Dt(1,this.accBeta2);n.forEach((o,i)=>{const c=E.registeredVariables[o],a=!1;this.accumulatedFirstMoment[i]==null&&(this.accumulatedFirstMoment[i]={originalName:`${o}/m`,variable:U(()=>ct(c).variable(a))}),this.accumulatedSecondMoment[i]==null&&(this.accumulatedSecondMoment[i]={originalName:`${o}/v`,variable:U(()=>ct(c).variable(a))});const l=Array.isArray(t)?t[i].tensor:t[o];if(l==null)return;const h=this.accumulatedFirstMoment[i].variable,u=this.accumulatedSecondMoment[i].variable,f=F(A(h,this.beta1),A(l,1-this.beta1)),d=F(A(u,this.beta2),A(bt(l),1-this.beta2)),g=at(f,r),p=at(d,s);h.assign(f),u.assign(d);const m=F(A(at(g,F(Ot(p),this.epsilon)),-this.learningRate),c);c.assign(m)}),this.accBeta1.assign(A(this.accBeta1,this.beta1)),this.accBeta2.assign(A(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&J(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedSecondMoment!=null&&J(this.accumulatedSecondMoment.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t),U(()=>{this.accBeta1.assign(yn(this.beta1,this.iterations_+1)),this.accBeta2.assign(yn(this.beta2,this.iterations_+1))});const n=t.length/2,r=!1;this.accumulatedFirstMoment=t.slice(0,n).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})),this.accumulatedSecondMoment=t.slice(n,n*2).map(s=>({originalName:s.name,variable:s.tensor.variable(r)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon)}}/**
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
 */class Ci extends Tt{static get className(){return"Adamax"}constructor(t,n,r,s=null,o=0){super(),this.learningRate=t,this.beta1=n,this.beta2=r,this.epsilon=s,this.decay=o,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],U(()=>{this.iteration=xt(0).variable(),this.accBeta1=xt(n).variable()}),s==null&&(this.epsilon=E.backend.epsilon())}applyGradients(t){const n=Array.isArray(t)?t.map(r=>r.name):Object.keys(t);U(()=>{const r=Dt(1,this.accBeta1),s=at(-this.learningRate,F(A(this.iteration,this.decay),1));n.forEach((o,i)=>{const c=E.registeredVariables[o],a=!1;this.accumulatedFirstMoment[i]==null&&(this.accumulatedFirstMoment[i]={originalName:`${o}/m`,variable:ct(c).variable(a)}),this.accumulatedWeightedInfNorm[i]==null&&(this.accumulatedWeightedInfNorm[i]={originalName:`${o}/v`,variable:ct(c).variable(a)});const l=Array.isArray(t)?t[i].tensor:t[o];if(l==null)return;const h=this.accumulatedFirstMoment[i].variable,u=this.accumulatedWeightedInfNorm[i].variable,f=F(A(h,this.beta1),A(l,1-this.beta1)),d=A(u,this.beta2),g=ho(l),p=Xo(d,g);h.assign(f),u.assign(p);const m=F(A(at(s,r),at(f,F(p,this.epsilon))),c);c.assign(m)}),this.iteration.assign(F(this.iteration,1)),this.accBeta1.assign(A(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&J(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedWeightedInfNorm!=null&&J(this.accumulatedWeightedInfNorm.map(t=>t.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(t){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon,n.decay)}}/**
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
 */class lr extends Tt{static get className(){return"SGD"}constructor(t){super(),this.learningRate=t,this.setLearningRate(t)}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const o=Array.isArray(t)?t[s].tensor:t[r];if(o==null)return;const i=E.registeredVariables[r];U(()=>{const c=F(A(this.c,o),i);i.assign(c)})}),this.incrementIterations()}setLearningRate(t){this.learningRate=t,this.c!=null&&this.c.dispose(),this.c=oo(xt(-t))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(t){if(t=await this.extractIterations(t),t.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(t,n){return new t(n.learningRate)}}/**
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
 */class Li extends lr{static get className(){return"Momentum"}constructor(t,n,r=!1){super(t),this.learningRate=t,this.momentum=n,this.useNesterov=r,this.accumulations=[],this.m=xt(this.momentum)}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const o=E.registeredVariables[r];this.accumulations[s]==null&&(this.accumulations[s]={originalName:`${r}/momentum`,variable:U(()=>ct(o).variable(!1))});const i=this.accumulations[s].variable,c=Array.isArray(t)?t[s].tensor:t[r];c!=null&&U(()=>{let a;const l=F(A(this.m,i),c);this.useNesterov?a=F(A(this.c,F(c,A(l,this.m))),o):a=F(A(this.c,l),o),i.assign(l),o.assign(a)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&J(this.accumulations.map(t=>t.variable))}setMomentum(t){this.momentum=t}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulations=t.map(r=>({originalName:r.name,variable:r.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(t,n){return new t(n.learningRate,n.momentum,n.useNesterov)}}/**
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
 */class Pi extends Tt{static get className(){return"RMSProp"}constructor(t,n=.9,r=0,s=null,o=!1){if(super(),this.learningRate=t,this.decay=n,this.momentum=r,this.epsilon=s,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=o,s==null&&(this.epsilon=E.backend.epsilon()),t==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(t){(Array.isArray(t)?t.map(r=>r.name):Object.keys(t)).forEach((r,s)=>{const o=E.registeredVariables[r],i=!1;this.accumulatedMeanSquares[s]==null&&(this.accumulatedMeanSquares[s]={originalName:`${r}/rms`,variable:U(()=>ct(o).variable(i))}),this.accumulatedMoments[s]==null&&(this.accumulatedMoments[s]={originalName:`${r}/momentum`,variable:U(()=>ct(o).variable(i))}),this.accumulatedMeanGrads[s]==null&&this.centered&&(this.accumulatedMeanGrads[s]={originalName:`${r}/mg`,variable:U(()=>ct(o).variable(i))});const c=Array.isArray(t)?t[s].tensor:t[r];if(c==null)return;const a=this.accumulatedMeanSquares[s].variable,l=this.accumulatedMoments[s].variable;U(()=>{const h=F(A(a,this.decay),A(bt(c),1-this.decay));if(this.centered){const u=this.accumulatedMeanGrads[s].variable,f=F(A(u,this.decay),A(c,1-this.decay)),d=at(A(c,this.learningRate),Ot(Dt(h,F(bt(f),this.epsilon)))),g=F(A(l,this.momentum),d);a.assign(h),u.assign(f),l.assign(g);const p=Dt(o,g);o.assign(p)}else{const u=F(A(a,this.decay),A(bt(c),1-this.decay)),f=F(A(l,this.momentum),at(A(c,this.learningRate),Ot(F(u,this.epsilon))));a.assign(u),l.assign(f);const d=Dt(o,f);o.assign(d)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&J(this.accumulatedMeanSquares.map(t=>t.variable)),this.accumulatedMeanGrads!=null&&this.centered&&J(this.accumulatedMeanGrads.map(t=>t.variable)),this.accumulatedMoments!=null&&J(this.accumulatedMoments.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&t.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=this.centered?t.length/3:t.length/2,r=!1;this.accumulatedMeanSquares=t.slice(0,n).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})),this.accumulatedMoments=t.slice(n,n*2).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})),this.centered&&(this.accumulatedMeanGrads=t.slice(n*2,n*3).map(s=>({originalName:s.name,variable:s.tensor.variable(r)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(t,n){return new t(n.learningRate,n.decay,n.momentum,n.epsilon,n.centered)}}/**
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
 */const Ui=[Ni,_i,Oi,Ci,Li,Pi,lr];function Gi(){for(const e of Ui)Di(e)}function Wi(e,t){const n=e.shape.length,r=t.shape.length;if(n<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${n}.`);if(r<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${r}.`);if(t.dtype!=="int32")throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[r-1]>n)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[r-1]} vs. ${n}`);if(V(e.shape)===0)throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${e.shape}.`);const s=t.shape,o=s[s.length-1];let i=1;for(let u=0;u<s.length-1;++u)i*=s[u];const c=e.shape,a=s.slice();a.pop();let l=1;for(let u=o;u<n;++u)l*=c[u],a.push(c[u]);const h=[...Vt(e.shape).map(u=>u/l),1].slice(0,o);return[a,i,l,h]}/**
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
 */const Pe=-2,zi=-1;function qi(e,t,n){const r=e.shape.length;x(r===t.length,()=>`Error in slice${r}D: Length of begin ${t} must match the rank of the array (${r}).`),x(r===n.length,()=>`Error in slice${r}D: Length of size ${n} must match the rank of the array (${r}).`);for(let s=0;s<r;++s)x(t[s]+n[s]<=e.shape[s],()=>`Error in slice${r}D: begin[${s}] + size[${s}] (${t[s]+n[s]}) would overflow input.shape[${s}] (${e.shape[s]})`)}function Vi(e){const t=[];let n=0;for(;e>0;)e&1&&t.push(n),e/=2,n++;return t}function ji(e,t,n){const r=[];for(let s=0;s<e.length;s++)r[s]=Math.ceil((t[s]-e[s])/n[s]);return r}function ur(e,t,n,r){const s=[...e];for(let o=s.length;o<r.length;o++)s.push(1);for(let o=0;o<n;o++)o===0?s[t]=1:(s.splice(t,0,1),s.pop());return s}function fr(e,t,n){return n<=e?n:n-(t-1)}function hr(e,t){const n=[];for(let r=0;r<e;r++)n.push(t+r);return n}function Hi(e,t,n,r,s,o,i,c,a){const l=e.length;let h=new Array(l),u=new Array(l),f=new Array(l);if(t.length&&n>0){const d=t[0],g=n+1;h=dr(i,d,g,r,e),u=gr(c,d,g,s,e),f=ur(o,d,g,e)}else for(let d=0;d<l;d++)h[d]=mr(i,r,o,e,d,a),u[d]=yr(c,s,o,e,d,a),f[d]=pr(o,d,a);return{begin:h,end:u,strides:f}}function dr(e,t,n,r,s){const o=[...s],i=hr(n,t);for(let c=0;c<o.length;c++)if(i.indexOf(c)>-1)o[c]=0;else{const a=fr(t,n,c);let l=r[a];e&1<<a&&(l=0),o[c]=l}return o}function gr(e,t,n,r,s){const o=[...s],i=hr(n,t);for(let c=0;c<o.length;c++)if(i.indexOf(c)>-1)o[c]=Number.MAX_SAFE_INTEGER;else{const a=fr(t,n,c);let l=r[a];e&1<<a&&(l=Number.MAX_SAFE_INTEGER),o[c]=l}for(let c=0;c<o.length;c++){const a=s[c];o[c]<0&&(o[c]+=a),o[c]=Zt(0,o[c],s[c])}return o}function pr(e,t,n){let r=e[t];return(n&1<<t||r==null)&&(r=1),r}function mr(e,t,n,r,s,o){let i=t[s];const c=n[s]||1;(e&1<<s||o&1<<s||i==null)&&(c>0?i=Number.MIN_SAFE_INTEGER:i=Number.MAX_SAFE_INTEGER);const a=r[s];return i<0&&(i+=a),i=Zt(0,i,a-1),i}function yr(e,t,n,r,s,o){let i=t[s];const c=n[s]||1;(e&1<<s||o&1<<s||i==null)&&(c>0?i=Number.MAX_SAFE_INTEGER:i=Number.MIN_SAFE_INTEGER);const a=r[s];return i<0&&(i+=a),c>0?i=Zt(0,i,a):i=Zt(-1,i,a-1),i}function Ki(e,t,n){let r=n.length;for(let s=0;s<n.length;s++)if(n[s]>1){r=s;break}for(let s=r+1;s<n.length;s++)if(t[s]>0||n[s]!==e[s])return!1;return!0}function Xi(e,t){let n=e.length>0?e[e.length-1]:1;for(let r=0;r<e.length-1;r++)n+=e[r]*t[r];return n}function Zi(e,t,n){let r;const s=e.shape.length;typeof t=="number"?r=[t,...new Array(s-1).fill(0)]:t.length<s?r=t.concat(new Array(s-t.length).fill(0)):r=t.slice(),r.forEach(i=>{x(i!==-1,()=>"slice() does not support negative begin indexing.")});let o;return n==null?o=new Array(s).fill(-1):typeof n=="number"?o=[n,...new Array(s-1).fill(-1)]:n.length<s?o=n.concat(new Array(s-n.length).fill(-1)):o=n,o=o.map((i,c)=>i>=0?i:(x(i===-1,()=>`Negative size values should be exactly -1 but got ${i} for the slice() size at index ${c}.`),e.shape[c]-r[c])),[r,o]}function Ji(e,t,n,r,s,o,i,c,a){let l;if(r==null?(l=new Array(t.length),l.fill(1)):l=r,i!=null&&i&i-1)throw new Error("Multiple ellipses in slice is not allowed.");let h=!1;const u={dims:l.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:n.slice(),strides:l.slice(),beginMask:s,endMask:o,ellipsisMask:i,newAxisMask:c,shrinkAxisMask:a};for(let w=0;w<u.dims;w++)h&&1<<w&c&&u.numAddAxisAfterEllipsis++,1<<w&i&&(h=!0);h||(u.ellipsisMask|=1<<u.dims,u.dims++);const f={dims:e.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};Yi(u,f);let d=!0,g=!0,p=!0;const m=[],b=[];for(let w=0;w<e.length;++w){if(f.strides[w]===0)throw Error(`strides[${w}] must be non-zero`);const S=!!(f.shrinkAxisMask&1<<w),I=e[w];if(I===-1){m.push(S?1:-1);continue}const T=[f.beginMask&1<<w,f.endMask&1<<w],$=[f.strides[w]>0?0:-1,f.strides[w]>0?I:I-1];if(S&&f.strides[w]<=0)throw Error("only stride 1 allowed on non-range indexing.");p=p&&f.strides[w]===1;const M=!!(f.beginMask&1<<w&&f.endMask&1<<w);if(f.beginValid&&f.endValid){if(S){const W=f.begin[w]<0?I+f.begin[w]:f.begin[w];if(f.begin[w]=W,f.end[w]=f.begin[w]+1,W<0||W>=I)throw Error(`slice index ${f.begin[w]} of dimension ${w} out of bounds.`)}else f.begin[w]=bn(f.begin[w],0,f.strides[w],I,T,$),f.end[w]=bn(f.end[w],1,f.strides[w],I,T,$);const C=f.strides[w]===1&&f.begin[w]===0&&f.end[w]===I;d=d&&C,g=g&&(w===0&&f.strides[w]===1||C)}else d=d&&f.strides[w]===1&&M,g=g&&(w===0&&f.strides[w]===1||M);let B,G=!1;if(f.beginValid&&f.endValid?(B=f.end[w]-f.begin[w],G=!0):S?(B=1,G=!0):M&&I>=0&&(f.strides[w]<0?B=-I:B=I,G=!0),G){let C;B===0||B<0!=f.strides[w]<0?C=0:C=Math.trunc(B/f.strides[w])+(B%f.strides[w]!==0?1:0),m.push(C)}else m.push(-1)}for(let w=0;w<f.finalShapeGatherIndices.length;++w){const S=f.finalShapeGatherIndices[w];S>=0?b.push(m[S]):S===Pe&&b.push(1)}return{finalShapeSparse:b.filter((w,S)=>f.finalShapeGatherIndices[S]!==Pe),finalShape:b,isIdentity:d,sliceDim0:g,isSimpleSlice:p,begin:f.begin,end:f.end,strides:f.strides}}function Yi(e,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let n=0;t.beginValid=e.begin!=null,t.endValid=e.end!=null,t.begin=new Array(t.dims),t.end=new Array(t.dims),t.strides=new Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=new Array(t.dims);for(let r=0;r<e.dims;r++)if(1<<r&e.ellipsisMask){const s=Math.min(t.dims-(e.dims-r)+1+e.numAddAxisAfterEllipsis,t.dims);for(;n<s;n++)t.begin[n]=0,t.end[n]=0,t.strides[n]=1,t.beginMask|=1<<n,t.endMask|=1<<n,t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[n]=r}else if(1<<r&e.newAxisMask)t.finalShapeGatherIndices.push(Pe),t.finalShapeGatherIndicesSparse.push(-1);else{if(n===t.begin.length)throw Error(`Index out of range using input dim ${n}; input has only ${t.dims} dims, ${t.begin.length}.`);e.begin!=null&&(t.begin[n]=e.begin[r]),e.end!=null&&(t.end[n]=e.end[r]),t.strides[n]=e.strides[r],e.beginMask&1<<r&&(t.beginMask|=1<<n),e.endMask&1<<r&&(t.endMask|=1<<n),e.shrinkAxisMask&1<<r?(t.finalShapeGatherIndices.push(zi),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<n):(t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(r)),t.inputShapeGatherIndicesSparse[n]=r,n++}}function bn(e,t,n,r,s,o){if(s[t])return n>0?o[t]:o[t+1&1];{const i=e<0?r+e:e;return i<o[0]?o[0]:i>o[1]?o[1]:i}}const Qi=Object.freeze(Object.defineProperty({__proto__:null,assertParamsValid:qi,computeFlatOffset:Xi,computeOutShape:ji,getNormalizedAxes:Hi,isSliceContinous:Ki,maskToAxes:Vi,parseSliceParams:Zi,sliceInfo:Ji,startForAxis:mr,startIndicesWithElidedDims:dr,stopForAxis:yr,stopIndicesWithElidedDims:gr,stridesForAxis:pr,stridesWithElidedDims:ur},Symbol.toStringTag,{value:"Module"}));/**
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
 */function ta(e,t){const n=e[0].length;e.forEach((s,o)=>{x(s.length===n,()=>`Error in concat${n}D: rank of tensors[${o}] must be the same as the rank of the rest (${n})`)}),x(t>=0&&t<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);const r=e[0];e.forEach((s,o)=>{for(let i=0;i<n;i++)x(i===t||s[i]===r[i],()=>`Error in concat${n}D: Shape of tensors[${o}] (${s}) does not match the shape of the rest (${r}) along the non-concatenated axis ${o}.`)})}function ea(e,t){const n=e[0].slice();for(let r=1;r<e.length;r++)n[t]+=e[r][t];return n}/**
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
 */var it;(function(e){e[e.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",e[e.VALUE_ROWIDS=1]="VALUE_ROWIDS",e[e.ROW_LENGTHS=2]="ROW_LENGTHS",e[e.ROW_SPLITS=3]="ROW_SPLITS",e[e.ROW_LIMITS=4]="ROW_LIMITS",e[e.ROW_STARTS=5]="ROW_STARTS"})(it||(it={}));function na(e,t,n){let r=new Array;if(n==null&&t==null)return r;if(t==null)for(;r.length<e+n.length;)r.push(-1);else r=t.slice();if(n==null)return r;if(e+n.length!==r.length)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${e+n.length}, but shape.rank = ${r.length}`);for(let s=1;s<n.length;++s){const o=n[s],i=r[r.length-n.length+s],c=r[i];if(o>=0)if(c>=0){if(c!==o)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${s+e}] = ${o} but shape[${s+e}] = ${c}`)}else r[i]=o}return r}function ra(e){const t={FIRST_DIM_SIZE:it.FIRST_DIM_SIZE,VALUE_ROWIDS:it.VALUE_ROWIDS,ROW_LENGTHS:it.ROW_LENGTHS,ROW_SPLITS:it.ROW_SPLITS,ROW_LIMITS:it.ROW_LIMITS,ROW_STARTS:it.ROW_STARTS},n=[];for(const r of e)if(r in t)n.push(t[r]);else break;return n}function sa(e){return e.length===0?0:e[0]===it.FIRST_DIM_SIZE?e.length-1:e.length}function oa(e,t){if(e==null||t==null)return;const n=e.length,r=t.length;if(n>=r)throw new Error(`defaultValue.shape=${e} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${n} must be less than ragged tensor input flatValues.rank = ${r})`);for(let s=0;s<Math.min(n,r-1);++s){const o=e[s],i=t[s+1];if(o>=0&&i>=0&&o!==1&&o!==i)throw new Error(`defaultValue.shape=${e}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${s-e.length}] = ${o} but ragged tensor input.flatValues.shape[${s-e.length}] = ${i}`)}}/**
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
 */const Qe=30;function ia(e){return e<=Qe?e:me(e,Math.floor(Math.sqrt(e)))}/**
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
 */function aa(e,t,n){const r=n*(typeof e=="number"?e:e[0]),s=t*(typeof e=="number"?e:e[1]);return[r,s]}/**
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
 */function ca(e,t,n,r=!0){let s=[];if(r)s=s.concat(t.slice(0)),s.push(e[0]/n),s=s.concat(e.slice(1));else{s=s.concat(e[0]);const o=t.length;for(let i=0;i<o;++i)s=s.concat([e[i+1]/t[i],t[i]]);s=s.concat(e.slice(o+1))}return s}function la(e,t,n=!0){const r=[];if(n){r.push(t);for(let s=t+1;s<e;++s)s<=2*t?(r.push(s),r.push(s-(t+1))):r.push(s)}else{const s=[],o=[];for(let i=1;i<e;++i)i>=t*2+1||i%2===1?o.push(i):s.push(i);r.push(...s),r.push(0),r.push(...o)}return r}function ua(e,t,n,r=!0){const s=[];r?s.push(e[0]/n):s.push(e[0]*n);for(let o=1;o<e.length;++o)o<=t.length?r?s.push(t[o-1]*e[o]):s.push(e[o]/t[o-1]):s.push(e[o]);return s}function fa(e,t){const n=[0];for(let r=0;r<t;++r)n.push(e[r][0]);return n}function ha(e,t,n){const r=e.slice(0,1);for(let s=0;s<n;++s)r.push(e[s+1]-t[s][0]-t[s][1]);return r}/**
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
 */const da=1.7580993408473768,ga=1.0507009873554805;/**
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
 */const pa=.3275911,ma=.254829592,ya=-.284496736,wa=1.421413741,ba=-1.453152027,Sa=1.061405429;/**
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
 */function Ea(e,t){if(e.length!==t.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${e.length}, imag: ${t.length}.`);const n=new Float32Array(e.length*2);for(let r=0;r<n.length;r+=2)n[r]=e[r/2],n[r+1]=t[r/2];return n}function xa(e){const t=new Float32Array(e.length/2),n=new Float32Array(e.length/2);for(let r=0;r<e.length;r+=2)t[r/2]=e[r],n[r/2]=e[r+1];return{real:t,imag:n}}function Ia(e){const t=Math.ceil(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let s=0;s<e.length;s+=4)n[Math.floor(s/4)]=e[s],r[Math.floor(s/4)]=e[s+1];return{real:n,imag:r}}function Aa(e){const t=Math.floor(e.length/4),n=new Float32Array(t),r=new Float32Array(t);for(let s=2;s<e.length;s+=4)n[Math.floor(s/4)]=e[s],r[Math.floor(s/4)]=e[s+1];return{real:n,imag:r}}function ka(e,t){const n=e[t*2],r=e[t*2+1];return{real:n,imag:r}}function va(e,t,n,r){e[r*2]=t,e[r*2+1]=n}function Ta(e,t){const n=new Float32Array(e/2),r=new Float32Array(e/2);for(let s=0;s<Math.ceil(e/2);s++){const o=(t?2:-2)*Math.PI*(s/e);n[s]=Math.cos(o),r[s]=Math.sin(o)}return{real:n,imag:r}}function $a(e,t,n){const r=(n?2:-2)*Math.PI*(e/t),s=Math.cos(r),o=Math.sin(r);return{real:s,imag:o}}/**
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
 */const fe="->",Ma=/->/g,Sn=",",En="...";function Fa(e,t){e=e.replace(/\s/g,"");const n=(e.length-e.replace(Ma,"").length)/fe.length;if(n<1)throw new Error("Equations without an arrow are not supported.");if(n>1)throw new Error(`Equation must contain exactly one arrow ("${fe}").`);const[r,s]=e.split(fe);x(r.indexOf(En)===-1,()=>`The ellipsis notation ("${En}") is not supported yet.`);const o=r.split(Sn),i=o.length;if(t!==i)throw new Error(`Expected ${i} input tensors, received ${t}`);if(i>2)throw new Error("Support for more than 2 input tensors is not implemented yet.");const c=[];for(let f=0;f<s.length;++f){const d=s[f];if(!o.some(g=>g.indexOf(d)!==-1))throw new Error(`Output subscripts contain the label ${d} not present in the input subscripts.`);c.indexOf(d)===-1&&c.push(d)}for(let f=0;f<r.length;++f){const d=r[f];c.indexOf(d)===-1&&d!==Sn&&c.push(d)}const a=new Array(o.length);for(let f=0;f<i;++f){if(new Set(o[f].split("")).size!==o[f].length)throw new Error(`Found duplicate axes in input component ${o[f]}. Support for duplicate axes in input is not implemented yet.`);a[f]=[];for(let d=0;d<o[f].length;++d)a[f].push(c.indexOf(o[f][d]))}const l=c.length,h=s.length,u=[];for(let f=h;f<l;++f)u.push(f);return{allDims:c,summedDims:u,idDims:a}}function Ra(e,t){let n=new Array(e);n.fill(-1);for(let s=0;s<t.length;++s)n[t[s]]=s;const r=[];for(let s=0;s<e;++s)n[s]===-1&&r.push(s);return n=n.filter(s=>s!==-1),{permutationIndices:n,expandDims:r}}function Ba(e,t,n){const r=new Array(e);for(let s=0;s<n.length;++s){const o=n[s].shape;for(let i=0;i<t[s].length;++i)r[t[s][i]]===void 0?r[t[s][i]]=o[i]:x(r[t[s][i]]===o[i],()=>`Expected dimension ${r[t[s][i]]} at axis ${i} of input shaped ${JSON.stringify(o)}, but got dimension ${o[i]}`)}}function Da(e,t){const n=e,r=[];let s=0;e.length===0&&n.push(-1),s=e.length+1;for(let i=0;i<s;++i)r.push([]);const o=[];for(let i=0;i<n.length;++i){const c=n[i],a=_a(t,c);for(const l of a)o.indexOf(l)===-1&&(r[i].push(l),o.push(l))}return{path:n,steps:r}}function Na(e){return e.every((t,n)=>t===n)}function _a(e,t){const n=[];for(let r=0;r<e.length;++r)(e[r].length===0||e[r].indexOf(t)!==-1||t===-1)&&n.push(r);return n}function Oa(e,t,n=0){let r=[];if(typeof t=="number")x(e.shape[n]%t===0,()=>"Number of splits must evenly divide the axis."),r=new Array(t).fill(e.shape[n]/t);else{const s=t.reduce((i,c)=>(c===-1&&(i+=1),i),0);x(s<=1,()=>"There should be only one negative value in split array.");const o=t.indexOf(-1);if(o!==-1){const i=t.reduce((c,a)=>a>0?c+a:c);t[o]=e.shape[n]-i}x(e.shape[n]===t.reduce((i,c)=>i+c),()=>"The sum of sizes must match the size of the axis dimension."),r=t}return r}/**
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
 */function Ca(e){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${e}`}function La(e,t){return`indices(${e}, 0) is invalid: ${t} < 0`}function Pa(e,t,n){return`indices(${e}, 0) is invalid: ${t} >= ${n}`}/**
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
 */function Ua(e,t){return`only one output dimension may be -1, not both ${e} and ${t}`}function Ga(e,t){return`size ${e} must be non-negative, not ${t}`}function Wa(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function za(e,t){const n=V(e),r=V(t);return`Input to reshape is a SparseTensor with ${n}
  dense values, but the requested shape requires a multiple of ${r}. inputShape=${e} outputShape= ${t}`}function qa(e,t){const n=V(e),r=V(t);return`Input to reshape is a tensor with ${n} dense values, but the requested shape has ${r}. inputShape=${e} outputShape=${t}`}/**
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
 */function Va(){return"segment ids must be >= 0"}function ja(){return"segment ids are not increasing"}function Ha(e,t){return`Segment id ${e} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function Ka(e,t,n){return`Bad: indices[${e}] == ${t} out of range [0, ${n})`}/**
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
 */function Xa(e,t){let n=!1,r;for(e<=Qe?(r=e,n=!0):r=me(e,Math.floor(Math.sqrt(e)));!n;)r>t||r===e?n=!0:r=me(e,r+1);return r}function Za(e,t,n){const r=[],s=e.length;for(let o=0;o<s;o++)o!==t?r.push(e[o]):r.push(n);return r}function Ja(e,t,n,r){const s=t.shape.length,o=e.shape.length;if(r!==0&&(r<-s||r>s))throw new Error(`Expect batchDims in the range of [-${s}, ${s}], but got ${r}`);if(r<0&&(r+=s),r>o)throw new Error(`batchDims (${r}) must be less than rank(x) (
    ${o}).`);if(n<r)throw new Error(`batchDims (${r}) must be less than or equal to axis (${n}).`);for(let u=0;u<r;++u)if(e.shape[u]!==t.shape[u])throw new Error(`x.shape[${u}]: ${e.shape[u]} should be equal to indices.shape[${u}]: ${t.shape[u]}.`);const i=e.shape[n],c=[];let a=1,l=1,h=1;for(let u=0;u<r;++u)c.push(e.shape[u]),a*=e.shape[u];for(let u=r;u<n;u++)c.push(e.shape[u]),l*=e.shape[u];for(let u=r;u<s;u++)c.push(t.shape[u]);for(let u=n+1;u<o;u++)c.push(e.shape[u]),h*=e.shape[u];return{batchSize:a,sliceSize:h,outerSize:l,dimSize:i,outputShape:c}}const Ya=Object.freeze(Object.defineProperty({__proto__:null,collectGatherOpShapeInfo:Ja,computeOutShape:Za,segOpComputeOptimalWindowSize:Xa},Symbol.toStringTag,{value:"Module"}));/**
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
 */function Qa(e){try{return e.map(t=>Ee(t))}catch(t){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${t}`)}}function tc(e){return e.map(t=>je(t))}const Uf=Object.freeze(Object.defineProperty({__proto__:null,ERF_A1:ma,ERF_A2:ya,ERF_A3:wa,ERF_A4:ba,ERF_A5:Sa,ERF_P:pa,PARALLELIZE_THRESHOLD:Qe,get RowPartitionType(){return it},SELU_SCALE:ga,SELU_SCALEALPHA:da,applyActivation:Ai,assertAndGetBroadcastShape:ir,assertAxesAreInnerMostDims:_o,assertParamsConsistent:ta,assignToTypedArray:va,axesAreInnerMostDims:Je,calculateShapes:Ei,checkEinsumDimSizes:Ba,checkPadOnDimRoundingMode:Io,combineLocations:ar,combineRaggedTensorToTensorShapes:na,complexWithEvenIndex:Ia,complexWithOddIndex:Aa,computeConv2DInfo:Xe,computeConv3DInfo:nr,computeDefaultPad:Ze,computeDilation2DInfo:go,computeOptimalWindowSize:ia,computeOutAndReduceShapes:Do,computeOutShape:ea,computePool2DInfo:po,computePool3DInfo:mo,convertConv2DDataFormat:rr,decodeEinsumEquation:Fa,eitherStridesOrDilationsAreOne:Eo,expandShapeToKeepDim:No,exponent:$a,exponents:Ta,fromStringArrayToUint8:tc,fromUint8ToStringArray:Qa,getAxesPermutation:Oo,getBroadcastDims:Mo,getComplexWithIndex:ka,getEinsumComputePath:Da,getEinsumPermutation:Ra,getFusedBiasGradient:Ii,getFusedDyActivation:xi,getImageCenter:aa,getInnerMostAxes:Lo,getPermuted:la,getRaggedRank:sa,getReductionAxes:or,getReshaped:ca,getReshapedPermuted:ua,getRowPartitionTypesHelper:ra,getSliceBeginCoords:fa,getSliceSize:ha,getSparseFillEmptyRowsIndicesDenseShapeMismatch:Ca,getSparseFillEmptyRowsNegativeIndexErrorMessage:La,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:Pa,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:Wa,getSparseReshapeInputOutputMismatchErrorMessage:qa,getSparseReshapeInputOutputMultipleErrorMessage:za,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:Ua,getSparseReshapeNegativeOutputDimErrorMessage:Ga,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:Ka,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:Va,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:ja,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:Ha,getUndoAxesPermutation:Co,isIdentityPermutation:Na,log:ss,mergeRealAndImagArrays:Ea,prepareAndValidate:Wi,prepareSplitSize:Oa,segment_util:Ya,shouldFuse:ki,slice_util:Qi,splitRealAndImagArrays:xa,stridesOrDilationsArePositive:xo,tupleValuesAreOne:Re,upcastType:He,validateDefaultValueShape:oa,validateInput:Si,validateUpdateShape:cr,warn:ft},Symbol.toStringTag,{value:"Module"}));/**
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
 */Gi();export{xi as $,Sc as A,Lc as B,Dt as C,sl as D,E,at as F,yn as G,Su as H,Cu as I,Sl as J,hc as K,Fl as L,ql as M,le as N,ki as O,Ai as P,Eo as Q,au as R,xu as S,Q as T,Io as U,Xe as V,kt as W,ir as X,Nf as Y,rf as Z,Re as _,x as a,$c as a$,Ii as a0,L as a1,wf as a2,pf as a3,mf as a4,v as a5,yf as a6,Hn as a7,pn as a8,gf as a9,ro as aA,pi as aB,yi as aC,Ot as aD,bt as aE,bi as aF,zo as aG,ct as aH,oo as aI,U as aJ,We as aK,J as aL,sc as aM,It as aN,ec as aO,ai as aP,Ff as aQ,wc as aR,bc as aS,Ec as aT,xc as aU,Ic as aV,Ac as aW,kc as aX,vc as aY,Tc as aZ,Mc as a_,bf as aa,vf as ab,df as ac,If as ad,Os as ae,xf as af,Af as ag,Tf as ah,Ef as ai,Sf as aj,kf as ak,en as al,Qu as am,Gt as an,er as ao,Fs as ap,ho as aq,Df as ar,ne as as,Bs as at,Bo as au,$o as av,co as aw,Vo as ax,Xo as ay,Jo as az,Ue as b,ru as b$,Fc as b0,Bc as b1,Gc as b2,Nc as b3,ku as b4,ju as b5,_c as b6,wl as b7,Oc as b8,Pc as b9,bl as bA,El as bB,xl as bC,Al as bD,kl as bE,vl as bF,Tl as bG,$l as bH,Ml as bI,de as bJ,Cl as bK,Rl as bL,Bl as bM,Jl as bN,Mu as bO,Dl as bP,Nl as bQ,_l as bR,Ul as bS,Wl as bT,Vl as bU,An as bV,Mr as bW,Hl as bX,Kl as bY,Xl as bZ,Yl as b_,Uc as ba,xo as bb,Wc as bc,qc as bd,Vc as be,Hc as bf,Kc as bg,Xc as bh,Zc as bi,Jc as bj,Qc as bk,tl as bl,el as bm,ol as bn,fl as bo,Iu as bp,ul as bq,Pl as br,jl as bs,Ir as bt,No as bu,hl as bv,dl as bw,gl as bx,es as by,yl as bz,k as c,Tn as c$,nu as c0,ou as c1,Fu as c2,po as c3,iu as c4,_f as c5,uu as c6,fu as c7,hu as c8,yu as c9,ml as cA,tf as cB,Ql as cC,Cf as cD,eu as cE,Pf as cF,tu as cG,Lf as cH,pu as cI,du as cJ,Ku as cK,Du as cL,Nu as cM,_u as cN,Ou as cO,Wu as cP,zu as cQ,qu as cR,Uu as cS,lr as cT,Li as cU,Pi as cV,Oi as cW,Ni as cX,Ci as cY,_i as cZ,Or as c_,wu as ca,bu as cb,Au as cc,$u as cd,vu as ce,Tu as cf,Bu as cg,pl as ch,Il as ci,Ru as cj,Lu as ck,fc as cl,su as cm,Gu as cn,Vu as co,Hu as cp,Zu as cq,Yu as cr,Ju as cs,Xu as ct,zc as cu,nf as cv,nl as cw,rl as cx,ef as cy,Yc as cz,F as d,Mo as d$,or as d0,Dc as d1,Rc as d2,Cc as d3,$n as d4,Lr as d5,jc as d6,Oo as d7,il as d8,al as d9,Yr as dA,Pu as dB,rs as dC,ts as dD,Qr as dE,ns as dF,of as dG,Is as dH,Qt as dI,Bf as dJ,Di as dK,Bi as dL,mt as dM,Mf as dN,$r as dO,Tt as dP,rc as dQ,Er as dR,nc as dS,$f as dT,ft as dU,Ge as dV,je as dW,Ea as dX,gc as dY,Ee as dZ,Vt as d_,Ur as da,ll as db,Wr as dc,Co as dd,Mn as de,zr as df,Ol as dg,Ll as dh,qr as di,zl as dj,Gl as dk,Do as dl,Vr as dm,jr as dn,Hr as dp,Pr as dq,Zr as dr,Kr as ds,Xr as dt,mu as du,gu as dv,da as dw,ga as dx,Jr as dy,Zi as dz,cl as e,Ta as e$,yc as e0,mc as e1,Cr as e2,dc as e3,Ve as e4,Qa as e5,xn as e6,cf as e7,Lo as e8,He as e9,mo as eA,ca as eB,la as eC,ua as eD,fa as eE,ha as eF,ta as eG,ea as eH,rr as eI,nr as eJ,go as eK,pc as eL,Fa as eM,Ba as eN,Da as eO,Ra as eP,Na as eQ,ma as eR,ya as eS,wa as eT,ba as eU,Sa as eV,pa as eW,ka as eX,xa as eY,Ia as eZ,Aa as e_,ra as ea,sa as eb,oa as ec,na as ed,it as ee,qi as ef,Ki as eg,Xi as eh,tc as ei,Ca as ej,La as ek,Pa as el,Ua as em,Ga as en,Wa as eo,za as ep,qa as eq,Va as er,ja as es,Ha as et,Ka as eu,af as ev,he as ew,Rf as ex,uc as ey,_o as ez,V as f,va as f0,$a as f1,Gr as f2,re as f3,Wi as f4,Ja as f5,aa as f6,Ei as f7,Oa as f8,Ji as f9,ji as fa,Zt as fb,sf as fc,cc as fd,Ht as fe,oc as ff,ff as fg,Uf as fh,lc as fi,Wt as fj,ic as fk,ge as fl,Ms as fm,ia as fn,lf as fo,Za as fp,Xa as fq,os as fr,we as fs,as as ft,hf as g,Zl as h,cu as i,lu as j,se as k,Me as l,A as m,Ss as n,O as o,ac as p,qn as q,sr as r,vo as s,jn as t,Eu as u,Si as v,Of as w,xr as x,uf as y,xt as z};
