'use client';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

const colors = ['#8b5cf6','#38bdf8','#34d399'];

export function Confidence({ values }:{ values:{name:string;value:number}[] }) {
  return <section className="panel"><div className="panel-head"><div><h3>Average confidence</h3><p>Model confidence across detected cells</p></div></div><div className="conf-list">{values.map(x=><div className="conf" key={x.name}><b>{x.name}</b><span><i className={x.name.toLowerCase()} style={{width:`${Math.max(0,Math.min(100,x.value))}%`}}/></span><strong>{x.value.toFixed(1)}%</strong></div>)}</div></section>
}

export function Distribution({ data }:{ data:{name:string;value:number;count:number}[] }) {
  const total = data.reduce((a,b)=>a+b.count,0);
  return <section className="panel"><div className="panel-head"><div><h3>Cell distribution</h3><p>Detected cell population</p></div></div><div className="dist"><div className="chart"><ResponsiveContainer width="100%" height={180}><PieChart><Pie data={data} dataKey="value" innerRadius={53} outerRadius={73} paddingAngle={3}>{data.map((x,i)=><Cell key={x.name} fill={colors[i%colors.length]}/>)}</Pie><Tooltip/></PieChart></ResponsiveContainer><b>{total.toLocaleString()}<small>cells</small></b></div><div className="legend">{data.map((x,i)=><p key={x.name}><span><i style={{background:colors[i%colors.length]}}/>{x.name}</span><b>{x.value.toFixed(1)}%</b></p>)}</div></div></section>
}
