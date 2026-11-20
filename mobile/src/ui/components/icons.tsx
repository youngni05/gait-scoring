/**
 * 선 아이콘 모음. 디자인은 Font Awesome을 CDN으로 불러왔지만,
 * 앱 초안은 외부 아이콘 서버 없이 동작하도록 같은 의미의 SVG를 직접 그린다.
 * 색은 글자색(currentColor)을 따른다.
 */
import type { ReactNode } from 'react';

type P = { size?: number; className?: string };

function Icon({ size = 20, className, sw = 2, children }: P & { sw?: number; children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={sw}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
    >
      {children}
    </svg>
  );
}

export const WalkIcon = (p: P) => (
  <Icon {...p} sw={2.4}>
    <circle cx="13" cy="4" r="2" />
    <path d="M10 22l2-6 3 3v3M8 12l2-4 4 1 3 3M12 9l-1 7" />
  </Icon>
);
export const ArrowLeftIcon = (p: P) => <Icon {...p} sw={2.4}><path d="M19 12H5M12 19l-7-7 7-7" /></Icon>;
export const CloseIcon = (p: P) => <Icon {...p} sw={2.4}><path d="M18 6L6 18M6 6l12 12" /></Icon>;
export const ChevronRightIcon = (p: P) => <Icon {...p} sw={2.4}><path d="M9 6l6 6-6 6" /></Icon>;
export const EyeIcon = (p: P) => (
  <Icon {...p}><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></Icon>
);
export const EyeOffIcon = (p: P) => (
  <Icon {...p}>
    <path d="M10.6 5.1A10 10 0 0112 5c6.5 0 10 7 10 7a17 17 0 01-2.2 3.2M6.6 6.6A17 17 0 002 12s3.5 7 10 7a9.7 9.7 0 005.4-1.6M3 3l18 18M9.9 9.9a3 3 0 004.2 4.2" />
  </Icon>
);
export const AlertCircleIcon = (p: P) => (
  <Icon {...p}><circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" /></Icon>
);
export const TriangleAlertIcon = (p: P) => (
  <Icon {...p}><path d="M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0zM12 9v4M12 17h.01" /></Icon>
);
export const InfoIcon = (p: P) => (
  <Icon {...p}><circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" /></Icon>
);
export const SearchIcon = (p: P) => <Icon {...p} sw={2.4}><circle cx="11" cy="11" r="7" /><path d="M20 20l-4-4" /></Icon>;
export const ClearIcon = (p: P) => (
  <Icon {...p}><circle cx="12" cy="12" r="10" /><path d="M15 9l-6 6M9 9l6 6" /></Icon>
);
export const ClockIcon = (p: P) => <Icon {...p}><circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" /></Icon>;
export const HistoryIcon = (p: P) => (
  <Icon {...p}><path d="M3 12a9 9 0 109-9 9.7 9.7 0 00-6.7 2.7L3 8M3 3v5h5M12 7v5l4 2" /></Icon>
);
export const UserIcon = (p: P) => <Icon {...p}><circle cx="12" cy="8" r="4" /><path d="M4 21c0-4 4-6 8-6s8 2 8 6" /></Icon>;
export const UserPlusIcon = (p: P) => (
  <Icon {...p}><circle cx="9" cy="8" r="4" /><path d="M2 21c0-4 3-6 7-6s7 2 7 6M19 8v6M16 11h6" /></Icon>
);
export const UsersIcon = (p: P) => (
  <Icon {...p}><circle cx="9" cy="8" r="4" /><path d="M2 21c0-4 3-6 7-6s7 2 7 6M17 11a3 3 0 000-6M22 21c0-3-2-5-4-5.5" /></Icon>
);
export const PlusIcon = (p: P) => <Icon {...p} sw={3}><path d="M12 5v14M5 12h14" /></Icon>;
export const PenIcon = (p: P) => <Icon {...p}><path d="M17 3l4 4L8 20H4v-4z" /></Icon>;
export const VideoIcon = (p: P) => (
  <Icon {...p}><rect x="2" y="6" width="14" height="12" rx="2" /><path d="M16 10l6-3v10l-6-3" /></Icon>
);
export const FilmIcon = (p: P) => (
  <Icon {...p}><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M7 3v18M17 3v18M3 8h4M3 16h4M17 8h4M17 16h4M3 12h18" /></Icon>
);
export const FolderOpenIcon = (p: P) => (
  <Icon {...p}><path d="M6 14l1.5-3a2 2 0 011.8-1H20a1 1 0 011 1.3l-1.6 6.2A2 2 0 0117.5 19H4a2 2 0 01-2-2V5a2 2 0 012-2h4l2 3h7a2 2 0 012 2v2" /></Icon>
);
export const CalendarIcon = (p: P) => (
  <Icon {...p}><rect x="3" y="4" width="18" height="18" rx="2" /><path d="M16 2v4M8 2v4M3 10h18" /></Icon>
);
export const CalculatorIcon = (p: P) => (
  <Icon {...p}><rect x="4" y="2" width="16" height="20" rx="2" /><path d="M8 6h8M8 11h.01M12 11h.01M16 11h.01M8 15h.01M12 15h.01M16 15h.01M8 19h8" /></Icon>
);
export const PlayIcon = (p: P) => (
  <svg width={p.size ?? 20} height={p.size ?? 20} viewBox="0 0 24 24" aria-hidden="true" className={p.className}>
    <path d="M7 4.5v15a1 1 0 001.5.9l12-7.5a1 1 0 000-1.8l-12-7.5A1 1 0 007 4.5z" fill="currentColor" />
  </svg>
);
export const PauseIcon = (p: P) => (
  <svg width={p.size ?? 20} height={p.size ?? 20} viewBox="0 0 24 24" aria-hidden="true" className={p.className}>
    <rect x="6" y="4" width="4" height="16" rx="1" fill="currentColor" />
    <rect x="14" y="4" width="4" height="16" rx="1" fill="currentColor" />
  </svg>
);
export const RotateIcon = (p: P) => (
  <Icon {...p}><path d="M21 12a9 9 0 11-3-6.7L21 8M21 3v5h-5" /></Icon>
);
export const RefreshIcon = (p: P) => (
  <Icon {...p}><path d="M3 12a9 9 0 0115-6.7L21 8M21 3v5h-5M21 12a9 9 0 01-15 6.7L3 16M3 21v-5h5" /></Icon>
);
export const ResetIcon = (p: P) => (
  <Icon {...p}><path d="M3 12a9 9 0 103-6.7L3 8M3 3v5h5" /></Icon>
);
export const CheckIcon = (p: P) => <Icon {...p} sw={3}><path d="M5 12l5 5L20 7" /></Icon>;
export const CheckCircleIcon = (p: P) => (
  <Icon {...p}><circle cx="12" cy="12" r="10" /><path d="M8 12l3 3 5-6" /></Icon>
);
export const CircleIcon = (p: P) => <Icon {...p}><circle cx="12" cy="12" r="10" /></Icon>;
export const SpinnerIcon = (p: P) => (
  <Icon {...p} sw={2.6} className={`spin ${p.className ?? ''}`}><path d="M21 12a9 9 0 11-6.2-8.6" /></Icon>
);
export const FileIcon = (p: P) => (
  <Icon {...p}><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6M8 13h8M8 17h8" /></Icon>
);
export const DownloadIcon = (p: P) => (
  <Icon {...p} sw={2.4}><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" /></Icon>
);
export const ExpandIcon = (p: P) => (
  <Icon {...p} sw={2.4}><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" /></Icon>
);
export const SortIcon = (p: P) => <Icon {...p}><path d="M3 6h13M3 12h9M3 18h5M18 8v12M15 17l3 3 3-3" /></Icon>;
export const ChartLineIcon = (p: P) => <Icon {...p}><path d="M3 3v18h18M7 15l4-4 3 3 5-6" /></Icon>;
export const HandIcon = (p: P) => (
  <Icon {...p}><path d="M9 11V5a2 2 0 014 0v5M13 10V4a2 2 0 014 0v7M17 9a2 2 0 014 0v5a8 8 0 01-8 8h-1a7 7 0 01-5.6-2.8L3.5 15a2 2 0 013-2.6L9 15" /></Icon>
);
export const PhoneIcon = (p: P) => (
  <Icon {...p}><rect x="6" y="2" width="12" height="20" rx="2" /><path d="M11 18h2" /></Icon>
);
export const SlidersIcon = (p: P) => (
  <Icon {...p}><path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6" /></Icon>
);
