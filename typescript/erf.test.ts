/**
 * Locks the erf / standard-normal-CDF port to the exact values produced by
 * scipy (scipy.special.erf and scipy.stats.norm.cdf), the functions used by the
 * Python reference runtime and the metric-producing evaluation code. This is
 * what keeps the TypeScript/web percentile math in parity with Python.
 */

import { describe, it, expect } from "vitest";
import { erf, standardNormalCDF } from "./erf.js";

// [z, scipy.special.erf(z), scipy.stats.norm.cdf(z)] — generated once from scipy.
const REFERENCE: [number, number, number][] = [
  [-6, -1.0, 9.865876450376944e-10],
  [-4, -0.9999999845827421, 3.167124183311986e-5],
  [-3, -0.9999779095030014, 0.001349898031630093],
  [-2.5, -0.999593047982555, 0.006209665325776134],
  [-2, -0.9953222650189527, 0.022750131948179198],
  [-1.5, -0.9661051464753108, 0.06680720126885806],
  [-1, -0.8427007929497148, 0.15865525393145707],
  [-0.5, -0.5204998778130465, 0.3085375387259869],
  [-0.25, -0.2763263901682369, 0.4012936743170763],
  [-0.1, -0.1124629160182849, 0.460172162722971],
  [0, 0.0, 0.5],
  [0.1, 0.1124629160182849, 0.539827837277029],
  [0.25, 0.2763263901682369, 0.5987063256829237],
  [0.5, 0.5204998778130465, 0.6914624612740131],
  [1, 0.8427007929497148, 0.8413447460685429],
  [1.5, 0.9661051464753108, 0.9331927987311419],
  [2, 0.9953222650189527, 0.9772498680518208],
  [2.5, 0.999593047982555, 0.9937903346742238],
  [3, 0.9999779095030014, 0.9986501019683699],
  [4, 0.9999999845827421, 0.9999683287581669],
  [6, 1.0, 0.9999999990134123],
  [0.0123, 0.013878363865855803, 0.5049068663219913],
  [-0.0123, -0.013878363865855803, 0.49509313367800867],
  [1.959963984540054, 0.9944254033192156, 0.975],
  [-1.959963984540054, -0.9944254033192156, 0.024999999999999998],
  [3.090232, 0.9999875894498337, 0.9989999989691049],
  [0.6745, 0.6598591796279205, 0.7500032571363009],
  // Sub-2^-28 |x| branch (erf(x) ≈ x + efx*x): locks the efx constant.
  [1e-10, 1.1283791670955126e-10, 0.5000000000398942],
  [-1e-10, -1.1283791670955126e-10, 0.49999999996010575],
];

// Tight: the fdlibm port matches scipy to within ~1 ULP.
const TOL = 1e-12;

describe("erf", () => {
  it("matches scipy.special.erf to < 1e-12", () => {
    for (const [z, expected] of REFERENCE) {
      expect(Math.abs(erf(z) - expected)).toBeLessThan(TOL);
    }
  });

  it("is odd: erf(-x) == -erf(x)", () => {
    for (const z of [0.3, 1.1, 2.4, 4.2]) {
      expect(erf(-z)).toBeCloseTo(-erf(z), 15);
    }
  });

  it("erf(NaN) is NaN", () => {
    expect(Number.isNaN(erf(NaN))).toBe(true);
  });
});

describe("standardNormalCDF", () => {
  it("matches scipy.stats.norm.cdf to < 1e-12", () => {
    for (const [z, , expected] of REFERENCE) {
      expect(Math.abs(standardNormalCDF(z) - expected)).toBeLessThan(TOL);
    }
  });

  it("is bounded to [0, 1]", () => {
    for (const z of [-40, -6, 0, 6, 40]) {
      const p = standardNormalCDF(z);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });
});
