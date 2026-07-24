const numberFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 1,
});

export function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat(undefined, {
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value);
}

export function formatDateTime(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value));
}

export function formatRelativeTime(value: string): string {
  const timestamp = new Date(value).getTime();
  const seconds = Math.round((timestamp - Date.now()) / 1000);
  const abs = Math.abs(seconds);

  const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ['year', 31536000],
    ['month', 2592000],
    ['week', 604800],
    ['day', 86400],
    ['hour', 3600],
    ['minute', 60],
  ];

  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });

  for (const [unit, unitSeconds] of units) {
    if (abs >= unitSeconds) {
      return formatter.format(Math.round(seconds / unitSeconds), unit);
    }
  }

  return formatter.format(seconds, 'second');
}

export function formatDuration(seconds: number): string {
  if (seconds < 1) {
    return `${Math.round(seconds * 1000)} ms`;
  }

  return `${numberFormatter.format(seconds)} s`;
}

export function estimateTitle(content: string): string {
  const firstLine = content.trim().split('\n').find(Boolean) ?? 'New chat';
  const normalized = firstLine.replace(/\s+/g, ' ');
  return normalized.length > 52 ? `${normalized.slice(0, 49)}...` : normalized;
}
