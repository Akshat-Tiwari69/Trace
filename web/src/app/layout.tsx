import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";

import "./globals.css";

const publicSans = localFont({
  src: "./fonts/public-sans-latin.woff2",
  weight: "100 900",
  style: "normal",
  variable: "--font-sans",
  display: "swap",
});

const fraunces = localFont({
  src: "./fonts/fraunces-latin.woff2",
  weight: "600",
  style: "normal",
  variable: "--font-display",
  display: "swap",
});

export const metadata: Metadata = {
  title: { default: "TRACE — Route Resilience Field Atlas", template: "%s — TRACE" },
  description: "Explore, stress-test, compare and recover an urban road network from satellite-derived evidence.",
  applicationName: "TRACE",
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  colorScheme: "light",
  themeColor: "#0b1413",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${publicSans.variable} ${fraunces.variable}`}>
      <body>{children}</body>
    </html>
  );
}
