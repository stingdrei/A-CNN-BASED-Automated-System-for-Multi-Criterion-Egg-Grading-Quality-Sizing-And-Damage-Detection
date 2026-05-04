import axios, { AxiosInstance, AxiosError } from 'axios';

const API_BASE_URL = '/api/v1';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          const isAuthPage = window.location.pathname.startsWith('/login') || 
                           window.location.pathname.startsWith('/register');
          
          if (!isAuthPage) {
            localStorage.removeItem('token');
            // Notify Zustand store to clear in-memory state (avoids circular dependency)
            window.dispatchEvent(new CustomEvent('auth:logout'));
            // Redirect to login
            window.location.href = '/login';
          }
          // For auth pages (login/register), just reject without clearing state
          return Promise.reject(error);
        }
        return Promise.reject(error);
      }
    );
  }

  // Auth
  async register(email: string, username: string, password: string) {
    const response = await this.client.post('/auth/register', { email, username, password });
    return response.data;
  }

  async login(email: string, password: string) {
    const response = await this.client.post('/auth/login', { email, password });
    return response.data;
  }

  async getMe() {
    const response = await this.client.get('/auth/me');
    return response.data;
  }

  // Predictions
  async uploadFile(file: File, confidenceThreshold: number = 0.75, saveAnnotated: boolean = true) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('confidence_threshold', confidenceThreshold.toString());
    formData.append('save_annotated', saveAnnotated.toString());

    const response = await this.client.post('/predictions/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  async getPredictions(skip: number = 0, limit: number = 20) {
    const response = await this.client.get('/predictions', { params: { skip, limit } });
    return response.data;
  }

  async getPrediction(id: number) {
    const response = await this.client.get(`/predictions/${id}`);
    return response.data;
  }

  async deletePrediction(id: number) {
    const response = await this.client.delete(`/predictions/${id}`);
    return response.data;
  }

  async downloadAnnotated(id: number) {
    const response = await this.client.get(`/predictions/${id}/download`, {
      responseType: 'blob',
    });
    return response.data;
  }

  // Dashboard
  async getDashboardStats() {
    const response = await this.client.get('/dashboard/stats');
    return response.data;
  }

  async healthCheck() {
    const response = await this.client.get('/dashboard/health');
    return response.data;
  }

  async getGradeDistribution() {
    const response = await this.client.get('/dashboard/grades');
    return response.data;
  }

  async getSettings() {
    const response = await this.client.get('/settings');
    return response.data;
  }

  async updateSettings(settings: { mm_per_pixel: number }) {
    const response = await this.client.put('/settings', settings);
    return response.data;
  }
}

export const apiClient = new ApiClient();