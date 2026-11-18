import { createContext } from 'react';
import type { ApiClient } from './client';
export const ApiSession = createContext<{ client: ApiClient; onSaved: () => void } | null>(null);
