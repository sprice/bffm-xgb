export type Likert = 1 | 2 | 3 | 4 | 5;

export interface Item {
  id: string;
  text: string;
  domain: string;
  domainCode: string;
  isReverseKeyed: boolean;
}

export interface QuantileValues {
  q05: number;
  q50: number;
  q95: number;
}

export interface DomainResult {
  raw: QuantileValues;
  percentile: QuantileValues;
}

export type PredictionResult = Record<string, DomainResult>;

export const DOMAIN_LABELS: Record<string, string> = {
  ext: "Extraversion",
  agr: "Agreeableness",
  csn: "Conscientiousness",
  est: "Emotional Stability",
  opn: "Intellect / Imagination",
};

export const DOMAIN_ORDER = ["ext", "agr", "csn", "est", "opn"] as const;

export const LIKERT_LABELS: Record<Likert, string> = {
  1: "Very Inaccurate",
  2: "Moderately Inaccurate",
  3: "Neither Accurate Nor Inaccurate",
  4: "Moderately Accurate",
  5: "Very Accurate",
};
