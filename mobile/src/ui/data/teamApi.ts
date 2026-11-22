import { ApiClient } from '../../api/client';

const tokenKey = 'team-access-token';
export const teamApi = new ApiClient(
  import.meta.env.VITE_TEAM_API_BASE === '/api'
    ? location.origin + '/api'
    : import.meta.env.VITE_TEAM_API_BASE || location.origin + '/api',
  () => sessionStorage.getItem(tokenKey) ?? '',
  async (url, init) => {
    const response = await fetch(url, init);
    const sent = new Headers(init?.headers).get('Authorization');
    if (response.status === 401 && sent && sent === 'Bearer ' + sessionStorage.getItem(tokenKey)) {
      sessionStorage.removeItem(tokenKey);
      window.dispatchEvent(new Event('team-session-expired'));
    }
    return response;
  },
);
