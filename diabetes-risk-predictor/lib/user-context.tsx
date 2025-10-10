"use client";

import React, { createContext, useContext, useEffect, useState } from 'react';
import { AuthService } from './auth';
import { UserSession, UserResponse } from './types';

interface UserContextType {
  user: UserResponse | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  signup: (username: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  updateUser: (user: UserResponse) => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export function UserProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const session = AuthService.getSession();
    if (session) {
      setUser(session.user);
    }
    setIsLoading(false);
  }, []);

  const login = async (username: string, password: string) => {
    try {
      const authResponse = await AuthService.login(username, password);
      AuthService.saveSession(authResponse);
      setUser(authResponse.user);
    } catch (error) {
      throw error;
    }
  };

  const signup = async (username: string, email: string, password: string) => {
    try {
      const authResponse = await AuthService.signup(username, email, password);
      AuthService.saveSession(authResponse);
      setUser(authResponse.user);
    } catch (error) {
      throw error;
    }
  };

  const logout = () => {
    AuthService.clearSession();
    setUser(null);
  };

  const updateUser = (updatedUser: UserResponse) => {
    setUser(updatedUser);
    if (typeof window !== 'undefined') {
      localStorage.setItem('user_data', JSON.stringify(updatedUser));
    }
  };

  const value: UserContextType = {
    user,
    isAuthenticated: !!user,
    isLoading,
    login,
    signup,
    logout,
    updateUser,
  };

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
}

export function useUser() {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
}
