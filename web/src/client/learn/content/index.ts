import type { Chapter } from "../types";

import { chapter01Orientation } from "./chapters/ch01-orientation";
import { chapter02Psychometrics } from "./chapters/ch02-psychometrics";
import { chapter03Stats } from "./chapters/ch03-stats";
import { chapter04Map } from "./chapters/ch04-map";
import { chapter05DataPipeline } from "./chapters/ch05-data-pipeline";
import { chapter06ItemAnalysis } from "./chapters/ch06-item-analysis";
import { chapter07Xgboost } from "./chapters/ch07-xgboost";
import { chapter08Tuning } from "./chapters/ch08-tuning";
import { chapter09Training } from "./chapters/ch09-training";
import { chapter10Evaluation } from "./chapters/ch10-eval";
import { chapter11Runtime } from "./chapters/ch11-runtime";
import { chapter12SafeChanges } from "./chapters/ch12-safe-changes";

export const chapters: Chapter[] = [
  chapter01Orientation,
  chapter02Psychometrics,
  chapter03Stats,
  chapter04Map,
  chapter05DataPipeline,
  chapter06ItemAnalysis,
  chapter07Xgboost,
  chapter08Tuning,
  chapter09Training,
  chapter10Evaluation,
  chapter11Runtime,
  chapter12SafeChanges,
].sort((a, b) => a.order - b.order);
