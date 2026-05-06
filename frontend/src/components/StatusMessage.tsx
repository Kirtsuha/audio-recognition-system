type StatusMessageProps = {
  title?: string;
  children: string;
  tone?: "error" | "info" | "success";
};

export function StatusMessage({ title, children, tone = "info" }: StatusMessageProps) {
  return (
    <div className={`status-message ${tone}`} role={tone === "error" ? "alert" : "status"}>
      {title && <strong>{title}</strong>}
      <span>{children}</span>
    </div>
  );
}
