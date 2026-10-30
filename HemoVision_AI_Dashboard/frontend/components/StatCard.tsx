import {Activity,CircleDot,Droplets,ScanSearch} from 'lucide-react';
const icons={RBC:Activity,WBC:ScanSearch,Platelet:Droplets,'Total Cells':CircleDot};
export default function StatCard({label,value,sub,tone}:{label:keyof typeof icons;value:number;sub:string;tone:string}){const Icon=icons[label];return <div className={'stat '+tone}><span className="stat-icon"><Icon size={18}/></span><div><small>{label}</small><strong>{value.toLocaleString()}</strong><em>{sub}</em></div></div>}
