export type Comparison = {
  id: string;
  createdAt: string;
  beforeUrl: string;
  afterUrl: string;
  scores: {
    correlation: number;
    chiSquare: number;
    bhattacharyya: number;
  };
  diffImages: Record<number, string>; // Stored as { threshold_value: image_url }
};
