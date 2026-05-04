import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { User, Token } from '../types';
import { apiClient } from '../api/client';

interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  error: string | null;
  
  login: (email: string, password: string) => Promise<boolean>;
  register: (email: string, username: string, password: string) => Promise<boolean>;
  logout: () => void;
  checkAuth: () => Promise<boolean>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          const data: Token = await apiClient.login(email, password);
          localStorage.setItem('token', data.access_token);
          set({ token: data.access_token, isLoading: false });
          return await get().checkAuth();
        } catch (error: unknown) {
          const message = error instanceof Error ? error.message : 'Login failed';
          set({ error: message, isLoading: false });
          return false;
        }
      },

      register: async (email: string, username: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          await apiClient.register(email, username, password);
          return await get().login(email, password);
        } catch (error: unknown) {
          const message = error instanceof Error ? error.message : 'Registration failed';
          set({ error: message, isLoading: false });
          return false;
        }
      },

      logout: () => {
        localStorage.removeItem('token');
        set({ user: null, token: null, error: null });
      },

      checkAuth: async () => {
        const token = localStorage.getItem('token');
        if (!token) {
          set({ token: null, user: null, isLoading: false });
          return false;
        }
        // Optimistically set token so UI doesn't flash
        set({ token, isLoading: true });
        try {
          const user = await apiClient.getMe();
          set({ user, isLoading: false });
          return true;
        } catch {
          // Token is invalid — clear everything
          localStorage.removeItem('token');
          set({ token: null, user: null, isLoading: false });
          return false;
        }
      },
    }),
    { name: 'auth-storage', partialize: (state) => ({ token: state.token }) }
  )
);

// Listen for 401 events from the API client to clear Zustand state
// This avoids a circular dependency (client.ts → authStore.ts → client.ts)
if (typeof window !== 'undefined') {
  window.addEventListener('auth:logout', () => {
    useAuthStore.getState().logout();
  });
}