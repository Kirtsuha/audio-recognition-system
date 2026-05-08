import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../api/client";
import type { UserProfile } from "../api/types";
import { useAuth } from "../auth/AuthContext";

const baseNavItems = [
  { to: "/recognition", label: "Recognition" },
  { to: "/search", label: "Search" },
  { to: "/history", label: "History" },
  { to: "/profile", label: "Profile" },
];

export function AppLayout() {
  const { logout } = useAuth();
  const [profile, setProfile] = useState<UserProfile | null>(null);

  useEffect(() => {
    let ignore = false;

    api.getProfile()
      .then((nextProfile) => {
        if (!ignore) {
          setProfile(nextProfile);
        }
      })
      .catch(() => {
        if (!ignore) {
          setProfile(null);
        }
      });

    return () => {
      ignore = true;
    };
  }, []);

  const navItems = profile?.role === "ROLE_ADMIN"
    ? [...baseNavItems, { to: "/admin", label: "Admin" }]
    : baseNavItems;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">MR</div>
          <div>
            <div className="brand-title">Music Recognition</div>
            <div className="brand-subtitle">User console</div>
          </div>
        </div>

        <nav className="nav-list" aria-label="Main navigation">
          {navItems.map((item) => (
            <NavLink key={item.to} to={item.to} className="nav-link">
              {item.label}
            </NavLink>
          ))}
        </nav>

        <button className="ghost-button sidebar-logout" onClick={logout}>
          Logout
        </button>
      </aside>

      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
