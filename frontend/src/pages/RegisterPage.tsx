import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { StatusMessage } from "../components/StatusMessage";

export function RegisterPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { setAuthenticatedToken } = useAuth();

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await api.register(username.trim(), password);
      if (response.accessToken) {
        setAuthenticatedToken(response.accessToken);
        navigate("/recognition", { replace: true });
      } else {
        navigate("/login", { replace: true });
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось создать аккаунт");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-panel">
        <div className="auth-heading">
          <div className="brand-mark">MR</div>
          <h1>Register</h1>
          <p>Создайте аккаунт для доступа к пользовательскому кабинету.</p>
        </div>

        <form className="form-stack" onSubmit={handleSubmit}>
          <label>
            Username
            <input
              autoComplete="username"
              value={username}
              minLength={3}
              onChange={(event) => setUsername(event.target.value)}
              required
            />
          </label>

          <label>
            Password
            <input
              autoComplete="new-password"
              type="password"
              value={password}
              minLength={4}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
          </label>

          {error && <StatusMessage tone="error">{error}</StatusMessage>}

          <button className="primary-button" disabled={loading}>
            {loading ? "Creating..." : "Register"}
          </button>
        </form>

        <p className="auth-footer">
          Уже есть аккаунт? <Link to="/login">Login</Link>
        </p>
      </section>
    </main>
  );
}
