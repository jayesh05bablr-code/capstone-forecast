"""TradingView-style intraday chart for the Live markets page (TradingView Lightweight Charts, bundled).

One self-contained HTML component: candles with a volume pane, VWAP and EMA 9/21, the opening range,
previous close and forecast open as price lines, an O/H/L/C/volume readout that follows the crosshair,
timeframe buttons (1, 3, 5, 15 minutes, resampled in the browser from the 1-minute bars) and indicator
toggles. Times are shown in IST. The library is shipped with the app (capstone/ui/vendor), so it does
not depend on a CDN.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

VENDOR = Path(__file__).with_name("vendor") / "lightweight-charts.standalone.production.js"
IST_OFFSET = 19800  # seconds; the chart's clock is UTC, so bars are shifted to read as IST

THEMES = {
    "dark": dict(bg="#0b0f17", panel="#111722", text="#c9d1dc", muted="#7d8796", grid="rgba(128,138,160,0.10)",
                 border="#232b3a", up="#089981", down="#f23645", upv="rgba(8,153,129,0.45)", downv="rgba(242,54,69,0.45)",
                 btn="#151b27", btnOn="#22304a", accent="#5aa9ff"),
    "light": dict(bg="#ffffff", panel="#f6f7f9", text="#1f2430", muted="#6b7280", grid="rgba(40,50,70,0.08)",
                  border="#e3e6ec", up="#089981", down="#f23645", upv="rgba(8,153,129,0.40)", downv="rgba(242,54,69,0.40)",
                  btn="#f1f3f6", btnOn="#dbe7fb", accent="#2a78d6"),
}


@lru_cache(maxsize=1)
def _library() -> str:
    return VENDOR.read_text(encoding="utf-8")


def bars_json(today: pd.DataFrame) -> list[dict]:
    """1-minute bars as [{t, o, h, l, c, v}] with t = IST wall-clock seconds (what the chart shows)."""
    d = today.sort_values("datetime").dropna(subset=["open", "high", "low", "close"])
    ts = pd.to_datetime(d["datetime"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    secs = (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    vol = d["volume"].fillna(0) if "volume" in d else pd.Series(0, index=d.index)
    return [{"t": int(t), "o": float(o), "h": float(h), "l": float(lo), "c": float(c), "v": float(v)}
            for t, o, h, lo, c, v in zip(secs, d["open"], d["high"], d["low"], d["close"], vol)]


def html(symbol: str, today: pd.DataFrame, prev_close: float | None = None, pred_open: float | None = None,
         opening_range: dict | None = None, dark: bool = True, height: int = 560, source: str = "") -> str:
    th = THEMES["dark" if dark else "light"]
    cfg = {"symbol": symbol, "bars": bars_json(today), "prev": prev_close or None, "fcst": pred_open or None,
           "or": opening_range or None, "theme": th, "height": height, "source": source}
    return TEMPLATE.replace("__LIB__", _library()).replace("__CFG__", json.dumps(cfg)).replace("__H__", str(height))


def render(symbol: str, today: pd.DataFrame, prev_close=None, pred_open=None, opening_range=None,
           dark: bool = True, height: int = 560, source: str = "") -> None:
    import streamlit.components.v1 as components
    components.html(html(symbol, today, prev_close, pred_open, opening_range, dark, height, source),
                    height=height + 48, scrolling=False)


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
*{box-sizing:border-box}html,body{margin:0;padding:0;background:transparent;font-family:Inter,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
#bar{display:flex;flex-wrap:wrap;align-items:center;gap:6px;padding:6px 2px 8px;font-size:12px}
#bar .grp{display:flex;gap:2px;padding:2px;border-radius:7px}
#bar button{border:0;border-radius:5px;padding:4px 9px;font-weight:600;font-size:12px;cursor:pointer;font-family:inherit}
#bar .sep{width:1px;height:18px;margin:0 4px}
#wrap{position:relative;border-radius:10px;overflow:hidden}
#legend{position:absolute;left:8px;top:6px;z-index:5;font-size:12px;line-height:1.55;pointer-events:none;padding:4px 8px;border-radius:6px;
  font-variant-numeric:tabular-nums}
#legend b{font-size:13px;margin-right:8px}#legend span{margin-right:10px}
#legend .ind span{margin-right:12px;font-weight:600}
#err{padding:24px;font-size:13px}
</style></head><body>
<div id="bar"></div><div id="wrap"><div id="legend"></div><div id="chart" style="height:__H__px"></div></div>
<script>__LIB__</script>
<script>
(function(){
const C=__CFG__, T=C.theme, LW=window.LightweightCharts;
const $=s=>document.querySelector(s);
document.body.style.color=T.text;
if(!LW||!C.bars.length){$('#chart').innerHTML='<div id="err">No intraday data to draw.</div>';return;}
const store={get(k,d){try{const v=localStorage.getItem('p8tv_'+k);return v===null?d:JSON.parse(v)}catch(e){return d}},
             set(k,v){try{localStorage.setItem('p8tv_'+k,JSON.stringify(v))}catch(e){}}};
const state={tf:store.get('tf',1),vwap:store.get('vwap',true),ema9:store.get('ema9',false),ema21:store.get('ema21',false),
             or:store.get('or',true),lv:store.get('lv',true)};
const fmt=x=>x==null||isNaN(x)?'—':x.toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2});
const fmtV=v=>v>=1e7?(v/1e7).toFixed(2)+' Cr':v>=1e5?(v/1e5).toFixed(2)+' L':v>=1e3?(v/1e3).toFixed(1)+' K':String(Math.round(v));
const wrap=$('#wrap');wrap.style.background=T.bg;$('#legend').style.background=T.bg+'cc';wrap.style.border='1px solid '+T.border;

const chart=LW.createChart($('#chart'),{autoSize:true,
  layout:{background:{type:'solid',color:T.bg},textColor:T.muted,fontSize:11,fontFamily:'Inter,Segoe UI,Roboto,sans-serif'},
  grid:{vertLines:{color:T.grid},horzLines:{color:T.grid}},
  crosshair:{mode:LW.CrosshairMode.Normal,vertLine:{color:T.muted,width:1,style:LW.LineStyle.Dashed,labelBackgroundColor:T.btnOn},
             horzLine:{color:T.muted,width:1,style:LW.LineStyle.Dashed,labelBackgroundColor:T.btnOn}},
  rightPriceScale:{borderColor:T.border,scaleMargins:{top:0.16,bottom:0.24}},
  timeScale:{borderColor:T.border,timeVisible:true,secondsVisible:false,rightOffset:4,barSpacing:7,minBarSpacing:1.5},
  localization:{priceFormatter:p=>'₹'+fmt(p)},
  handleScale:{axisPressedMouseMove:true},handleScroll:{vertTouchDrag:false}});
const candle=chart.addCandlestickSeries({upColor:T.up,downColor:T.down,borderUpColor:T.up,borderDownColor:T.down,
  wickUpColor:T.up,wickDownColor:T.down,priceLineVisible:true,lastValueVisible:true});
const vol=chart.addHistogramSeries({priceScaleId:'vol',priceFormat:{type:'volume'},lastValueVisible:false,priceLineVisible:false});
chart.priceScale('vol').applyOptions({scaleMargins:{top:0.80,bottom:0},visible:false});
const mk=(color,w)=>chart.addLineSeries({color,lineWidth:w,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
const vwapS=mk('#f2a93b',2), e9=mk('#5aa9ff',1), e21=mk('#c77dff',1);
let lines=[];

function resample(tf){
  if(tf===1)return C.bars.slice();
  const out=[],step=tf*60;let cur=null;
  for(const b of C.bars){
    const day=Math.floor(b.t/86400)*86400, open=day+9*3600+15*60;  // buckets start at 09:15 IST
    const k=open+Math.floor((b.t-open)/step)*step;
    if(!cur||cur.t!==k){if(cur)out.push(cur);cur={t:k,o:b.o,h:b.h,l:b.l,c:b.c,v:b.v};}
    else{cur.h=Math.max(cur.h,b.h);cur.l=Math.min(cur.l,b.l);cur.c=b.c;cur.v+=b.v;}
  }
  if(cur)out.push(cur);return out;
}
function ema(bars,n){const k=2/(n+1);let e=null;return bars.map(b=>{e=e===null?b.c:b.c*k+e*(1-k);return{time:b.t,value:e}})}
function vwapOf(bars){let pv=0,v=0,day=null;return bars.map(b=>{const d=Math.floor(b.t/86400);if(d!==day){day=d;pv=0;v=0}
  pv+=(b.h+b.l+b.c)/3*b.v;v+=b.v;return{time:b.t,value:v>0?pv/v:b.c}})}
let data=[],vw=[],m9=[],m21=[];
function draw(){
  data=resample(state.tf);
  candle.setData(data.map(b=>({time:b.t,open:b.o,high:b.h,low:b.l,close:b.c})));
  vol.setData(data.map(b=>({time:b.t,value:b.v,color:b.c>=b.o?T.upv:T.downv})));
  vw=vwapOf(data);m9=ema(data,9);m21=ema(data,21);
  vwapS.setData(state.vwap&&data.some(b=>b.v>0)?vw:[]);e9.setData(state.ema9?m9:[]);e21.setData(state.ema21?m21:[]);
  lines.forEach(l=>candle.removePriceLine(l));lines=[];
  const pl=(price,title,color,style)=>{if(price)lines.push(candle.createPriceLine({price,title,color,lineWidth:1,lineStyle:style,axisLabelVisible:true}))};
  if(state.lv){pl(C.prev,'Prev close',T.muted,LW.LineStyle.Dashed);pl(C.fcst,'Forecast open','#8b7bff',LW.LineStyle.Dashed);}
  if(state.or&&C.or){pl(C.or.high,'OR high','#6a8cff',LW.LineStyle.Dotted);pl(C.or.low,'OR low','#6a8cff',LW.LineStyle.Dotted);}
  legend(null);
}
function legend(i){
  const n=data.length;if(!n)return;const k=i==null||i<0||i>=n?n-1:i;const b=data[k];
  const ref=k>0?data[k-1].c:(C.prev||b.o),ch=(b.c-ref)/ref*100,col=b.c>=b.o?T.up:T.down;
  const d=new Date(b.t*1000),hh=String(d.getUTCHours()).padStart(2,'0'),mm=String(d.getUTCMinutes()).padStart(2,'0');
  let ind='';
  if(state.vwap&&vw[k])ind+=`<span style="color:#f2a93b">VWAP ${fmt(vw[k].value)}</span>`;
  if(state.ema9&&m9[k])ind+=`<span style="color:#5aa9ff">EMA 9 ${fmt(m9[k].value)}</span>`;
  if(state.ema21&&m21[k])ind+=`<span style="color:#c77dff">EMA 21 ${fmt(m21[k].value)}</span>`;
  const v=(l,x)=>`<span>${l} <b style="color:${col};font-size:12px;margin:0">${fmt(x)}</b></span>`;
  $('#legend').innerHTML=`<div><b style="color:${T.text}">${C.symbol}</b><span style="color:${T.muted}">${state.tf}m · ${hh}:${mm} IST</span>`+
    `<span style="color:${ch>=0?T.up:T.down};font-weight:600">${ch>=0?'+':''}${ch.toFixed(2)}%</span></div>`+
    `<div>${v('O',b.o)}${v('H',b.h)}${v('L',b.l)}${v('C',b.c)}<span style="color:${T.muted}">Vol ${fmtV(b.v)}</span></div>`+
    `<div class="ind">${ind}</div>`;
}
const index={};function reindex(){data.forEach((b,i)=>index[b.t]=i)}
chart.subscribeCrosshairMove(p=>{legend(p&&p.time!==undefined?index[p.time]:null)});

function btn(label,on,click,title){const e=document.createElement('button');e.textContent=label;e.title=title||'';
  const paint=v=>{e.style.background=v?T.btnOn:'transparent';e.style.color=v?T.text:T.muted};paint(on);
  e.onclick=()=>paint(click());return e}
function group(){const g=document.createElement('div');g.className='grp';g.style.background=T.btn;g.style.border='1px solid '+T.border;$('#bar').appendChild(g);return g}
const tfg=group(),tfButtons=[];
[1,3,5,15].forEach(tf=>{const b=btn(tf+'m',state.tf===tf,()=>{state.tf=tf;store.set('tf',tf);
  tfButtons.forEach(x=>x.b.style.background=x.tf===tf?T.btnOn:'transparent');tfButtons.forEach(x=>x.b.style.color=x.tf===tf?T.text:T.muted);
  draw();reindex();chart.timeScale().fitContent();return true},tf+'-minute candles');tfButtons.push({tf,b});tfg.appendChild(b)});
const ig=group();
[['vwap','VWAP','Volume-weighted average price since the open'],['ema9','EMA 9','9-bar exponential average'],
 ['ema21','EMA 21','21-bar exponential average'],['or','Opening range','High and low of the first minutes'],
 ['lv','Levels','Previous close and the model\'s forecast open']].forEach(([k,l,t])=>
  ig.appendChild(btn(l,state[k],()=>{state[k]=!state[k];store.set(k,state[k]);draw();return state[k]},t)));
const fg=group();fg.appendChild(btn('Fit',false,()=>{chart.timeScale().fitContent();return false},'Show the whole day'));
if(C.source){const s=document.createElement('span');s.style.cssText=`margin-left:auto;color:${T.muted};font-size:11px`;s.textContent=C.source;$('#bar').appendChild(s)}
draw();reindex();chart.timeScale().fitContent();
})();
</script></body></html>"""
