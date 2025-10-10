export function Footer() {
  return (
    <footer className="border-t border-border mt-10">
      <div className="mx-auto max-w-5xl px-4 py-8 text-center text-muted-foreground">
        © {new Date().getFullYear()} Diabetes Risk Predictor
      </div>
    </footer>
  )
}
