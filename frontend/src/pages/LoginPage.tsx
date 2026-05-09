import { FormEvent, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { StatusMessage } from "../components/StatusMessage";

type LocationState = {
  from?: {
    pathname?: string;
  };
};

export function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { setAuthenticatedToken } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as LocationState | null;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await api.login(username.trim(), password);
      setAuthenticatedToken(response.accessToken);
      navigate(state?.from?.pathname ?? "/recognition", { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось войти в систему");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-panel">
        <div className="auth-heading">
          <div className="brand-mark">MR</div>
          <h1>Login</h1>
          <p>Войдите, чтобы распознавать аудио и смотреть историю запросов.</p>
        </div>

        <form className="form-stack" onSubmit={handleSubmit}>
          <label>
            Username
            <input
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              required
            />
          </label>

          <label>
            Password
            <input
              autoComplete="current-password"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
          </label>

          {error && <StatusMessage tone="error">{error}</StatusMessage>}

          <button className="primary-button" disabled={loading}>
            {loading ? "Signing in..." : "Login"}
          </button>
        </form>

        <p className="auth-footer">
          Нет аккаунта? <Link to="/register">Register</Link>
        </p>
      </section>
    </main>
  );
}
