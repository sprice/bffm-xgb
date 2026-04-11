export interface ResourceLink {
  label: string;
  href: string;
  note?: string;
}

export interface Chapter {
  slug: string;
  order: number;
  title: string;
  kicker: string;
  summary: string;
  content: string;
  resources?: ResourceLink[];
  afterRender?: () => void;
}
