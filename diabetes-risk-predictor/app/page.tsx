// Landing Page
import Link from "next/link"
import Image from "next/image"

export default function HomePage() {
  return (
    <main className="min-h-dvh bg-background text-foreground">
      <header className="w-full border-b border-border">
        <div className="mx-auto max-w-5xl px-6 py-6 flex items-center justify-between">
          <div>
            <h1 className="text-balance text-3xl font-semibold text-primary">Sadar Diabetes</h1>
            <p className="text-sm text-muted-foreground mt-1">Prediksi Risiko Diabetes</p>
          </div>
          <nav className="flex items-center gap-3" aria-label="Primary">
            <Link href="/signup" className="rounded-lg bg-primary px-5 py-3 text-primary-foreground font-medium">
              Daftar
            </Link>
            <Link href="/login" className="rounded-lg bg-accent px-5 py-3 text-accent-foreground font-medium">
              Masuk
            </Link>
          </nav>
        </div>
      </header>

      <section className="mx-auto max-w-5xl px-6 py-12 grid md:grid-cols-2 gap-10 items-center">
        <div>
          <h2 className="text-balance text-4xl font-semibold mb-4">Periksa risiko diabetes Anda dengan mudah</h2>
          <p className="text-pretty mb-6">
            Periksa risiko diabetes Anda dengan mudah dan pelajari cara menjaga kesehatan. Alat sederhana kami dirancang untuk kenyamanan dan kejelasan, dengan teks besar, warna hangat, dan langkah-langkah yang jelas.
          </p>
          <div className="flex gap-4">
            <Link
              href="/signup"
              className="rounded-lg bg-primary px-6 py-4 text-primary-foreground font-semibold"
              aria-label="Mulai dengan mendaftar"
            >
              Mulai
            </Link>
            <Link
              href="/login"
              className="rounded-lg border border-border px-6 py-4 hover:bg-muted transition"
              aria-label="Masuk ke akun Anda"
            >
              Saya sudah memiliki akun
            </Link>
          </div>
        </div>
        <div className="flex justify-center">
          <Image
            src="/health-check-illustration.jpg"
            alt="Illustration of a friendly health check"
            width={360}
            height={260}
            className="rounded-xl border border-border"
          />
        </div>
      </section>
      <footer className="mt-10 border-t border-border">
        <div className="mx-auto max-w-5xl px-6 py-8 text-center text-muted-foreground">
          © {new Date().getFullYear()} Sadar Diabetes
        </div>
      </footer>
    </main>
  )
}
