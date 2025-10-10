import { UserSession, AuthResponse, ApiError } from './types';

const TOKEN_KEY = 'auth_token';
const USER_KEY = 'user_data';

export class AuthService {
  static async login(username: string, password: string): Promise<AuthResponse> {
    const response = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Login failed');
    }

    return data;
  }

  static async signup(username: string, email: string, password: string): Promise<AuthResponse> {
    const response = await fetch('/api/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Signup failed');
    }

    return data;
  }

  static saveSession(authResponse: AuthResponse): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem(TOKEN_KEY, authResponse.access_token);
      localStorage.setItem(USER_KEY, JSON.stringify(authResponse.user));
    }
  }

  static getSession(): UserSession | null {
    if (typeof window === 'undefined') return null;
    
    const token = localStorage.getItem(TOKEN_KEY);
    const userStr = localStorage.getItem(USER_KEY);
    
    if (!token || !userStr) return null;
    
    try {
      const user = JSON.parse(userStr);
      return { user, token };
    } catch {
      return null;
    }
  }

  static clearSession(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    }
  }

  static getAuthHeaders(): Record<string, string> {
    const session = this.getSession();
    return session ? { Authorization: `Bearer ${session.token}` } : {};
  }

  static isAuthenticated(): boolean {
    return this.getSession() !== null;
  }
}
