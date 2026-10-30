import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata={title:'HemoVision AI',description:'AI-assisted peripheral blood smear analysis'};
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>}
