import * as fs from 'node:fs';
import * as path from 'node:path';

interface MarketInput {
  id: string;
  name: string;
  slug: string;
  yes_token: string;
  start: string;
  prices: number[];
}

interface InputData {
  schema: string;
  competition: string;
  case: string;
  at: string;
  horizon_minutes?: number;
  quantiles: number[];
  event: string;
  markets: MarketInput[];
}

interface ForecastOutput {
  id: string;
  q: number[][];
}

interface OutputData {
  forecasts: ForecastOutput[];
}

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const TARGET_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
const SIGMA = 0.001;
const HGRID = [1, 2, 4, 7, 12, 20, 32, 50, 80, 120, 180, 240];

function clamp6(v: number): number {
  const clamped = Math.min(1, Math.max(0, v));
  const rounded = Math.round(clamped * 1000000) / 1000000;
  return rounded === 0 ? 0 : rounded;
}

function interpolateQuantiles(arr: number[], q: number): number {
  if (arr.length === 0) return 0;
  if (arr.length === 1) return arr[0];
  const pos = q * (arr.length - 1);
  const base = Math.floor(pos);
  const rest = pos - base;
  if (base + 1 < arr.length) {
    return arr[base] + rest * (arr[base + 1] - arr[base]);
  }
  return arr[arr.length - 1];
}

function main(): void {
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const files = fs.readdirSync(INPUT_DIR)
    .filter(f => f.endsWith('.in'))
    .sort();

  for (const file of files) {
    const inputPath = path.join(INPUT_DIR, file);
    const content = fs.readFileSync(inputPath, 'utf8');
    const data: InputData = JSON.parse(content);

    // Check quantiles requirement
    if (
      data.quantiles.length !== TARGET_QUANTILES.length ||
      !data.quantiles.every((q, idx) => Math.abs(q - TARGET_QUANTILES[idx]) < 1e-9)
    ) {
      throw new Error('Unsupported quantiles requested.');
    }

    const forecasts: ForecastOutput[] = [];

    // Compute empirical quantiles for each market and grid point
    const marketEmpiricals: Array<Array<number[] | null>> = [];

    for (const market of data.markets) {
      const p = market.prices;
      // Take last 720 prices or all if fewer
      const recentPrices = p.length > 720 ? p.slice(p.length - 720) : p;
      const n = recentPrices.length;

      const gridQuantiles: Array<number[] | null> = [];

      for (const h of HGRID) {
        const returns: number[] = [];
        for (let i = 0; i <= n - h - 1; i++) {
          returns.push(recentPrices[i + h] - recentPrices[i]);
        }

        if (returns.length < 20) {
          gridQuantiles.push(null);
        } else {
          returns.sort((a, b) => a - b);
          const qs = TARGET_QUANTILES.map(q => interpolateQuantiles(returns, q));
          gridQuantiles.push(qs);
        }
      }
      marketEmpiricals.push(gridQuantiles);
    }

    // Helper to interpolate empirical across sqrt(h)
    const getEmpiricalForH = (gridQList: Array<number[] | null>, h: number, j: number): number | null => {
      const targetSqrt = Math.sqrt(h);
      // Find surrounding grid points
      let lowerIdx = -1;
      let upperIdx = -1;

      for (let i = 0; i < HGRID.length; i++) {
        if (HGRID[i] <= h) {
          if (gridQList[i] !== null) lowerIdx = i;
        }
        if (HGRID[i] >= h) {
          if (gridQList[i] !== null && upperIdx === -1) {
            upperIdx = i;
            break;
          }
        }
      }

      if (lowerIdx === -1 && upperIdx === -1) return null;
      if (lowerIdx !== -1 && upperIdx === -1) {
        const qArr = gridQList[lowerIdx];
        return qArr ? qArr[j] : null;
      }
      if (lowerIdx === -1 && upperIdx !== -1) {
        const qArr = gridQList[upperIdx];
        return qArr ? qArr[j] : null;
      }

      const lGrid = HGRID[lowerIdx];
      const uGrid = HGRID[upperIdx];
      const lVal = gridQList[lowerIdx]![j];
      const uVal = gridQList[upperIdx]![j];

      if (lGrid === uGrid) return lVal;

      const lSqrt = Math.sqrt(lGrid);
      const uSqrt = Math.sqrt(uGrid);
      const weight = (targetSqrt - lSqrt) / (uSqrt - lSqrt);
      return lVal + weight * (uVal - lVal);
    };

    // Generate raw rows per market
    const rawRowsList: number[][][] = [];

    for (let mIdx = 0; mIdx < data.markets.length; mIdx++) {
      const market = data.markets[mIdx];
      const p = market.prices;
      const x = p[p.length - 1];
      const gridQList = marketEmpiricals[mIdx];
      const rows: number[][] = [];

      for (let h = 1; h <= (data.horizon_minutes ?? 240); h++) {
        const row: number[] = [];
        for (let j = 0; j < 5; j++) {
          const paramJ = Z[j] * SIGMA * Math.sqrt(h);
          const empJ = getEmpiricalForH(gridQList, h, j);
          const qJ = empJ !== null ? 0.4 * empJ + 0.6 * paramJ : paramJ;
          row.push(clamp6(x + qJ));
        }
        rows.push(row);
      }
      rawRowsList.push(rows);
    }

    // SUM rescale across markets
    let S = 0;
    for (const market of data.markets) {
      S += market.prices[market.prices.length - 1];
    }

    const scale = (S >= 0.85 && S <= 1.15) ? 1 / S : 1;

    for (let mIdx = 0; mIdx < data.markets.length; mIdx++) {
      const rows = rawRowsList[mIdx];
      for (let h = 0; h < 240; h++) {
        const row = rows[h];
        const median2 = clamp6(row[2] * scale);
        const shift = median2 - row[2];
        const shiftedRow = row.map(v => clamp6(v + shift));
        shiftedRow.sort((a, b) => a - b);
        rows[h] = shiftedRow;
      }

      forecasts.push({
        id: data.markets[mIdx].id,
        q: rows
      });
    }

    const outputData: OutputData = { forecasts };
    const outFileName = file.replace(/\.in$/, '.out');
    const outFilePath = path.join(OUTPUT_DIR, outFileName);
    fs.writeFileSync(outFilePath, JSON.stringify(outputData) + '\n', 'utf8');
  }
}

main();
