import {
  createContext,
  PropsWithChildren,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { clearStoredToken, getStoredToken, storeToken } from "./tokenStore";

type AuthContextValue = {
  token: string | null;
  isAuthenticated: boolean;
  setAuthenticatedToken: (token: string) => void;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: PropsWithChildren) {
  const [token, setToken] = useState<string | null>(() => getStoredToken());
  const navigate = useNavigate();
  const location = useLocation();

  const logout = useCallback(() => {
    clearStoredToken();
    setToken(null);
    navigate("/login", { replace: true });
  }, [navigate]);

  const setAuthenticatedToken = useCallback((nextToken: string) => {
    storeToken(nextToken);
    setToken(nextToken);
  }, []);

  useEffect(() => {
    const onExpired = () => {
      clearStoredToken();
      setToken(null);
      if (!location.pathname.startsWith("/login")) {
        navigate("/login", { replace: true });
      }
    };

    window.addEventListener("auth:expired", onExpired);
    return () => window.removeEventListener("auth:expired", onExpired);
  }, [location.pathname, navigate]);

  const value = useMemo(
    () => ({
      token,
      isAuthenticated: Boolean(token),
      setAuthenticatedToken,
      logout,
    }),
    [logout, setAuthenticatedToken, token],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
