import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { repoFacts } from "./content/data";

const repoRoot = resolve(import.meta.dirname, "..", "..", "..", "..");

function readJson<T>(relativePath: string): T {
  const fullPath = resolve(repoRoot, relativePath);
  return JSON.parse(readFileSync(fullPath, "utf-8")) as T;
}

const hasArtifacts = existsSync(
  resolve(repoRoot, "data/processed/load_metadata.json"),
);

describe.skipIf(!hasArtifacts)("repo facts used by the course", () => {
  it("matches cleaned-row and split metadata artifacts", () => {
    const loadMetadata = readJson<{
      row_counts: { n_valid: number };
    }>("data/processed/load_metadata.json");
    const splitMetadata = readJson<{
      total_valid: number;
      train_rows: number;
      val_rows: number;
      test_rows: number;
      stratification_scheme: string;
    }>("data/processed/ext_est/split_metadata.json");

    expect(repoFacts.totalValidRespondents).toBe(loadMetadata.row_counts.n_valid);
    expect(repoFacts.totalValidRespondents).toBe(splitMetadata.total_valid);
    expect(repoFacts.trainRows).toBe(splitMetadata.train_rows);
    expect(repoFacts.valRows).toBe(splitMetadata.val_rows);
    expect(repoFacts.testRows).toBe(splitMetadata.test_rows);
    expect(repoFacts.splitScheme).toBe(splitMetadata.stratification_scheme);
  });

  it("matches the first-item artifact", () => {
    const firstItem = readJson<{
      selected_item: { id: string; text: string; cross_domain_info: number };
    }>("data/processed/ext_est/first_item.json");

    expect(repoFacts.firstItemId).toBe(firstItem.selected_item.id);
    expect(repoFacts.firstItemText).toBe(firstItem.selected_item.text);
    expect(repoFacts.firstItemCrossDomainInfo).toBeCloseTo(
      firstItem.selected_item.cross_domain_info,
      4,
    );
  });

  it("matches the checked-in reference validation and baseline artifacts", () => {
    const researchSummary = readJson<{
      variants: {
        reference: {
          validation: {
            full_50: { pearson_r: number; mae: number; coverage_90: number };
            sparse_20: { pearson_r: number; mae: number; coverage_90: number };
          };
          baselines: {
            k20: {
              domain_balanced: { pearson_r: number; mae: number };
              mini_ipip: { pearson_r: number };
              adaptive_topk: { pearson_r: number };
              domain_constrained_adaptive: { pearson_r: number };
            };
          };
          simulation: {
            overall: { pearson_r: number; mae: number; coverage_90: number };
            domain_metrics: Record<string, { mean_items_used: number }>;
          };
        };
      };
    }>("artifacts/research_summary.json");

    const reference = researchSummary.variants.reference;

    expect(repoFacts.validation.full50R).toBeCloseTo(reference.validation.full_50.pearson_r, 4);
    expect(repoFacts.validation.full50Mae).toBeCloseTo(reference.validation.full_50.mae, 2);
    expect(repoFacts.validation.full50Coverage).toBeCloseTo(reference.validation.full_50.coverage_90, 4);
    expect(repoFacts.validation.sparse20R).toBeCloseTo(reference.validation.sparse_20.pearson_r, 4);
    expect(repoFacts.validation.sparse20Mae).toBeCloseTo(reference.validation.sparse_20.mae, 2);
    expect(repoFacts.validation.sparse20Coverage).toBeCloseTo(reference.validation.sparse_20.coverage_90, 4);

    expect(repoFacts.baselineK20.domainBalancedR).toBeCloseTo(
      reference.baselines.k20.domain_balanced.pearson_r,
      4,
    );
    expect(repoFacts.baselineK20.domainBalancedMae).toBeCloseTo(
      reference.baselines.k20.domain_balanced.mae,
      2,
    );
    expect(repoFacts.baselineK20.miniIpipStandaloneR).toBeCloseTo(
      reference.baselines.k20.mini_ipip.pearson_r,
      4,
    );
    expect(repoFacts.baselineK20.adaptiveTopKR).toBeCloseTo(
      reference.baselines.k20.adaptive_topk.pearson_r,
      4,
    );
    expect(repoFacts.baselineK20.constrainedAdaptiveR).toBeCloseTo(
      reference.baselines.k20.domain_constrained_adaptive.pearson_r,
      4,
    );

    expect(repoFacts.simulation.overallR).toBeCloseTo(reference.simulation.overall.pearson_r, 4);
    expect(repoFacts.simulation.overallMae).toBeCloseTo(reference.simulation.overall.mae, 2);
    expect(repoFacts.simulation.overallCoverage).toBeCloseTo(reference.simulation.overall.coverage_90, 4);
    expect(repoFacts.simulation.meanItemsPerDomain).toBe(
      reference.simulation.domain_metrics.ext.mean_items_used,
    );
  });

  it("matches the current tuned-params artifact and the ML-vs-averaging comparison artifact", () => {
    const tunedParams = readJson<{
      hyperparameters: {
        n_estimators: number;
        max_depth: number;
        learning_rate: number;
        reg_alpha: number;
        reg_lambda: number;
        subsample: number;
        colsample_bytree: number;
        min_child_weight: number;
      };
    }>("artifacts/tuned_params.json");
    const mlVsAveraging = readJson<{
      comparisons: Array<{
        method: string;
        n_items: number;
        ml_r: number;
        avg_r: number;
        delta_r: number;
      }>;
    }>("artifacts/variants/reference/ml_vs_averaging_comparison.json");

    expect(repoFacts.currentLockedParams).toEqual(tunedParams.hyperparameters);

    const domainBalanced = mlVsAveraging.comparisons.find(
      (row) => row.method === "domain_balanced" && row.n_items === 20,
    );
    const miniIpip = mlVsAveraging.comparisons.find(
      (row) => row.method === "mini_ipip" && row.n_items === 20,
    );

    expect(domainBalanced).toBeTruthy();
    expect(miniIpip).toBeTruthy();

    expect(repoFacts.mlVsAveragingK20.domainBalanced.mlR).toBeCloseTo(domainBalanced!.ml_r, 4);
    expect(repoFacts.mlVsAveragingK20.domainBalanced.avgR).toBeCloseTo(domainBalanced!.avg_r, 4);
    expect(repoFacts.mlVsAveragingK20.domainBalanced.deltaR).toBeCloseTo(domainBalanced!.delta_r, 4);

    expect(repoFacts.mlVsAveragingK20.miniIpip.mlR).toBeCloseTo(miniIpip!.ml_r, 4);
    expect(repoFacts.mlVsAveragingK20.miniIpip.avgR).toBeCloseTo(miniIpip!.avg_r, 4);
    expect(repoFacts.mlVsAveragingK20.miniIpip.deltaR).toBeCloseTo(miniIpip!.delta_r, 4);
  });
});
