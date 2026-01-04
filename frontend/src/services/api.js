import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://46.224.178.180:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add token to headers
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle token expiration
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.config.method?.toUpperCase(), response.config.url, response.status);
    return response;
  },
  (error) => {
    console.error('API Error:', {
      method: error.config?.method?.toUpperCase(),
      url: error.config?.url,
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      message: error.message
    });

    if (error.response?.status === 401) {
      console.warn('Authentication failed, redirecting to login');
      // Token expired or invalid
      localStorage.removeItem('token');
      // Redirect to login if on a protected route
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// API Functions
export const getDreamTeam = async (budget = null) => {
  console.log('🔄 [API] Fetching Dream Team from:', api.defaults.baseURL);
  console.log('🔄 [API] Request body:', budget ? { budget } : {});

  try {
    const requestBody = budget ? { budget } : {};
    const response = await api.post('/optimize/dream-team', requestBody);
    console.log('✅ [API] Dream Team response received:', response.status);
    return response.data;
  } catch (error) {
    console.error('❌ [API] Dream Team request failed');
    throw error;
  }
};

export const getPlayers = async () => {
  try {
    const response = await api.get('/players');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const getUserTeam = async () => {
  try {
    const response = await api.get('/user/team');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export default api;
