import { BrowserVideoSource } from './pose/browser-video-source';
import { MediaPipePoseExtractor } from './pose/mediapipe-extractor';
document.querySelector<HTMLButtonElement>('#run')!.onclick=async()=>{
 const out=document.querySelector('#result')!;out.textContent='';
 const records:unknown[]=[];
 const record=(method:string,url:string,body:boolean)=>{const u=new URL(url,location.href);records.push({method,protocol:u.protocol,origin:u.origin,path:u.protocol==='http:'||u.protocol==='https:'?u.pathname:'[inline data]',body});};
 const fetchOriginal=window.fetch,openOriginal=XMLHttpRequest.prototype.open,sendOriginal=XMLHttpRequest.prototype.send;
 const info=new WeakMap<XMLHttpRequest,{method:string;url:string}>();
 window.fetch=async(input,init)=>{record(init?.method??(input instanceof Request?input.method:'GET'),input instanceof Request?input.url:String(input),!!init?.body||(input instanceof Request&&input.body!==null));return fetchOriginal(input,init)};
 XMLHttpRequest.prototype.open=function(method:string,url:string|URL,async:boolean=true,username?:string|null,password?:string|null){info.set(this,{method,url:String(url)});return openOriginal.call(this,method,url,async,username,password)};
 XMLHttpRequest.prototype.send=function(body?:Document|XMLHttpRequestBodyInit|null){const i=info.get(this);if(i)record(i.method,i.url,body!=null);return sendOriginal.call(this,body)};
 let engine:MediaPipePoseExtractor|undefined;
 try{engine=await MediaPipePoseExtractor.create(new AbortController().signal);const c=document.createElement('canvas');c.width=c.height=256;await engine.extract({t:0,pixels:c,release(){}},new AbortController().signal);const file=document.querySelector<HTMLInputElement>('#file')!.files?.[0]; if(file){await engine.dispose();engine=await MediaPipePoseExtractor.create(new AbortController().signal);const session=await new BrowserVideoSource(document.createElement('div')).open(file,new AbortController().signal);try{for await(const frame of session.frames(new AbortController().signal)){await engine.extract(frame,new AbortController().signal);frame.release();break;}}finally{await session.dispose()}}out.textContent=JSON.stringify(records,null,2)}catch(e){out.textContent=String(e)+'\n'+JSON.stringify(records,null,2)}
 finally{await engine?.dispose();window.fetch=fetchOriginal;XMLHttpRequest.prototype.open=openOriginal;XMLHttpRequest.prototype.send=sendOriginal;}
};
