import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { UserProfile } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { StatusMessage } from "../components/StatusMessage";

export function ProfilePage() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(false);
  const { logout } = useAuth();

  const loadProfile = useCallback(async () => {
    setError("");
    setLoading(true);
    try {
      setProfile(await api.getProfile());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось загрузить профиль");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadProfile();
  }, [loadProfile]);

  async function deleteAccount() {
    if (!window.confirm("Удалить аккаунт? Это действие нельзя отменить.")) {
      return;
    }

    setError("");
    setDeleting(true);
    try {
      await api.deleteProfile();
      logout();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось удалить аккаунт");
    } finally {
      setDeleting(false);
    }
  }

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Profile</p>
          <h1>Профиль пользователя</h1>
        </div>
      </header>

      {error && <StatusMessage tone="error">{error}</StatusMessage>}

      <section className="panel profile-panel">
        {loading ? (
          <div className="loading-state">Загружаем профиль...</div>
        ) : profile ? (
          <>
            <dl className="detail-list">
              <div>
                <dt>ID</dt>
                <dd>{profile.id}</dd>
              </div>
              <div>
                <dt>Username</dt>
                <dd>{profile.username}</dd>
              </div>
              <div>
                <dt>Role</dt>
                <dd>{profile.role}</dd>
              </div>
            </dl>

            <div className="profile-actions">
              <button className="ghost-button" onClick={logout}>
                Logout
              </button>
              <button className="danger-button" disabled={deleting} onClick={deleteAccount}>
                {deleting ? "Deleting..." : "Delete account"}
              </button>
            </div>
          </>
        ) : (
          <div className="empty-state">Профиль не найден.</div>
        )}
      </section>
    </section>
  );
}
