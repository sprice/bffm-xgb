import { useCallback, useEffect, useMemo, useState } from "react";
import { ITEMS } from "../items";
import type { Item, Likert } from "../types";

const STORAGE_KEY = "bffm-session";

let sessionCleared = false;

interface SessionData {
  orderIds: string[];
  responses: Record<string, Likert>;
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const itemsById = new Map(ITEMS.map((item) => [item.id, item]));

const VALID_LIKERT = new Set<number>([1, 2, 3, 4, 5]);

function loadSession(): SessionData | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw) as SessionData & { activeIndex?: number };
    if (!Array.isArray(data.orderIds) || data.orderIds.length !== ITEMS.length) return null;
    if (data.orderIds.some((id) => !itemsById.has(id))) return null;
    for (const [key, value] of Object.entries(data.responses)) {
      if (!itemsById.has(key)) return null;
      if (!VALID_LIKERT.has(value)) return null;
    }
    return { orderIds: data.orderIds, responses: data.responses };
  } catch {
    return null;
  }
}

function saveSession(data: SessionData) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

export function clearSession() {
  sessionCleared = true;
  localStorage.removeItem(STORAGE_KEY);
}

export function hasSession(): boolean {
  return loadSession() !== null;
}

/**
 * Returns the 1-based question number to resume at, or "done" if all answered.
 * Creates a new session if none exists.
 */
export function getResumeTarget(): number | "done" {
  const existing = loadSession();
  if (existing) {
    const answered = new Set(Object.keys(existing.responses));
    if (answered.size >= existing.orderIds.length) return "done";
    const idx = existing.orderIds.findIndex((id) => !answered.has(id));
    return idx === -1 ? "done" : idx + 1;
  }
  // Create a new session
  const order = shuffle(ITEMS);
  saveSession({ orderIds: order.map((i) => i.id), responses: {} });
  return 1;
}

function initSession(): { order: Item[]; responses: Record<string, Likert> } {
  sessionCleared = false;
  const existing = loadSession();
  if (existing) {
    const order = existing.orderIds.map((id) => itemsById.get(id)!);
    return { order, responses: existing.responses };
  }
  const order = shuffle(ITEMS);
  saveSession({ orderIds: order.map((i) => i.id), responses: {} });
  return { order, responses: {} };
}

export function useAssessment() {
  const [{ order, responses: initResponses }] = useState(initSession);
  const [responses, setResponses] = useState<Record<string, Likert>>(initResponses);

  const totalItems = order.length;
  const answeredCount = Object.keys(responses).length;
  const allAnswered = answeredCount === totalItems;

  const orderIds = useMemo(() => order.map((i) => i.id), [order]);
  useEffect(() => {
    if (!sessionCleared) {
      saveSession({ orderIds, responses });
    }
  }, [orderIds, responses]);

  const answer = useCallback(
    (index: number, value: Likert) => {
      const item = order[index];
      if (item) {
        setResponses((prev) => ({ ...prev, [item.id]: value }));
      }
    },
    [order]
  );

  const rawResponses = responses as Record<string, number>;

  return {
    order,
    totalItems,
    answeredCount,
    allAnswered,
    responses,
    rawResponses,
    answer,
  };
}
