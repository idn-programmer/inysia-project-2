"use client"

import type React from "react"

import { useEffect, useMemo, useState } from "react"
import { useRouter } from "next/navigation"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { InputField, SelectField, ToggleField } from "@/components/input-field"
import { apiClient } from "@/lib/api"
import { useUser } from "@/lib/user-context"
import { PredictForm, PredictResponse } from "@/lib/types"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts'

export default function PredictPage() {
  const router = useRouter()
  const { user } = useUser()
  const [form, setForm] = useState<PredictForm>({
    age: "",
    gender: "Male",
    pulseRate: "",
    sbp: "",
    dbp: "",
    glucose: "",
    heightCm: "",
    weightKg: "",
    bmi: "",
    familyDiabetes: false,
    hypertensive: false,
    familyHypertension: false,
    cardiovascular: false,
    stroke: false,
  })
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const autoBmi = useMemo(() => {
    const h = Number(form.heightCm)
    const w = Number(form.weightKg)
    if (!h || !w) return ""
    const m = h / 100
    return Number((w / (m * m)).toFixed(1))
  }, [form.heightCm, form.weightKg])

  useEffect(() => {
    if (autoBmi && form.bmi === "") {
      setForm((f) => ({ ...f, bmi: autoBmi }))
    }
  }, [autoBmi, form.bmi])

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setIsLoading(true)
    setError("")
    
    try {
      const requestData = {
        age: form.age === "" ? undefined : Number(form.age),
        gender: form.gender,
        pulseRate: form.pulseRate === "" ? undefined : Number(form.pulseRate),
        sbp: form.sbp === "" ? undefined : Number(form.sbp),
        dbp: form.dbp === "" ? undefined : Number(form.dbp),
        glucose: form.glucose === "" ? undefined : Number(form.glucose),
        heightCm: form.heightCm === "" ? undefined : Number(form.heightCm),
        weightKg: form.weightKg === "" ? undefined : Number(form.weightKg),
        bmi: form.bmi === "" ? (autoBmi === "" ? undefined : Number(autoBmi)) : Number(form.bmi),
        familyDiabetes: form.familyDiabetes,
        hypertensive: form.hypertensive,
        familyHypertension: form.familyHypertension,
        cardiovascular: form.cardiovascular,
        stroke: form.stroke,
        userId: user?.id,
      }

      const data = await apiClient.predict(requestData)
      setResult(data)
      localStorage.setItem("lastRisk", String(data.risk))
      
      // Store prediction data for chatbot (excluding age and gender)
      const filteredShapValues = Object.fromEntries(
        Object.entries(data.shap_values || {}).filter(([key]) => key !== 'age' && key !== 'gender')
      )
      
      const predictionContext = {
        risk_score: data.risk,
        shap_values: filteredShapValues,
        features: {
          glucose: requestData.glucose,
          bmi: requestData.bmi,
          sbp: requestData.sbp,
          dbp: requestData.dbp,
          pulseRate: requestData.pulseRate,
          familyDiabetes: requestData.familyDiabetes,
          hypertensive: requestData.hypertensive,
          familyHypertension: requestData.familyHypertension,
          cardiovascular: requestData.cardiovascular,
          stroke: requestData.stroke,
          heightCm: requestData.heightCm,
          weightKg: requestData.weightKg,
        }
      }
      localStorage.setItem("predictionContext", JSON.stringify(predictionContext))
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediksi gagal")
    } finally {
      setIsLoading(false)
    }
  }

  const handleAskAI = () => {
    router.push('/chat')
  }

  const resultColor = result == null ? "" : result.risk < 33 ? "text-green-600" : result.risk < 66 ? "text-yellow-600" : "text-red-600"

  // Prepare SHAP chart data
  const shapChartData = useMemo(() => {
    if (!result || !result.shap_values) return []
    
    const featureLabels: Record<string, string> = {
      glucose: "Glukosa",
      bmi: "BMI",
      sbp: "Tekanan Darah Sistolik",
      dbp: "Tekanan Darah Diastolik",
      pulseRate: "Denyut Nadi",
      familyDiabetes: "Riwayat Diabetes Keluarga",
      hypertensive: "Hipertensi",
      familyHypertension: "Riwayat Hipertensi Keluarga",
      cardiovascular: "Kardiovaskular",
      stroke: "Stroke",
      heightCm: "Tinggi Badan",
      weightKg: "Berat Badan"
    }
    
    // Convert SHAP values to chart data, filter out age, gender, and very small values
    const data = Object.entries(result.shap_values)
      .filter(([key]) => key !== 'age' && key !== 'gender') // Exclude age and gender
      .map(([key, value]) => ({
        name: featureLabels[key] || key,
        contribution: Number(value.toFixed(3)),
        fill: value > 0 ? '#ef4444' : '#10b981' // red for positive, green for negative
      }))
      .filter(item => Math.abs(item.contribution) > 0.001) // Filter negligible values
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
      .slice(0, 7) // Top 7 factors
    
    return data
  }, [result])

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-4xl px-4 py-8">
        <h1 className="text-3xl font-semibold mb-6">Periksa Risiko Diabetes Anda</h1>
        {error && (
          <div className="mb-4 p-3 rounded-lg bg-red-100 text-red-700 border border-red-200">
            {error}
          </div>
        )}
        <form onSubmit={onSubmit} className="grid gap-5">
          <div className="grid gap-4 sm:grid-cols-2">
            <InputField
              label="Usia"
              name="age"
              type="number"
              value={form.age}
              onChange={(v) => setForm({ ...form, age: v === "" ? "" : Number(v) })}
              required
            />
            <SelectField
              label="Jenis Kelamin"
              name="gender"
              value={form.gender}
              onChange={(v) => setForm({ ...form, gender: v as "Male" | "Female" })}
              options={[
                { value: "Male", label: "Laki-laki" },
                { value: "Female", label: "Perempuan" },
              ]}
            />
            <InputField
              label="Denyut Nadi"
              name="pulseRate"
              type="number"
              value={form.pulseRate}
              onChange={(v) => setForm({ ...form, pulseRate: v === "" ? "" : Number(v) })}
            />
            <InputField
              label="Tekanan Darah Sistolik"
              name="sbp"
              type="number"
              value={form.sbp}
              onChange={(v) => setForm({ ...form, sbp: v === "" ? "" : Number(v) })}
            />
            <InputField
              label="Tekanan Darah Diastolik"
              name="dbp"
              type="number"
              value={form.dbp}
              onChange={(v) => setForm({ ...form, dbp: v === "" ? "" : Number(v) })}
            />
            <InputField
              label="Glukosa"
              name="glucose"
              type="number"
              value={form.glucose}
              onChange={(v) => setForm({ ...form, glucose: v === "" ? "" : Number(v) })}
            />
            <InputField
              label="Tinggi Badan (cm)"
              name="height"
              type="number"
              value={form.heightCm}
              onChange={(v) => setForm({ ...form, heightCm: v === "" ? "" : Number(v) })}
            />
            <InputField
              label="Berat Badan (kg)"
              name="weight"
              type="number"
              value={form.weightKg}
              onChange={(v) => setForm({ ...form, weightKg: v === "" ? "" : Number(v) })}
            />
            <InputField
              label="BMI"
              name="bmi"
              type="number"
              value={form.bmi === "" ? String(autoBmi) : String(form.bmi)}
              onChange={(v) => setForm({ ...form, bmi: v === "" ? "" : Number(v) })}
              hint="Otomatis dihitung dari tinggi dan berat badan; Anda dapat menyesuaikan jika diperlukan."
            />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <ToggleField
              label="Riwayat Diabetes Keluarga"
              name="familyDiabetes"
              checked={form.familyDiabetes}
              onChange={(v) => setForm({ ...form, familyDiabetes: v })}
            />
            <ToggleField
              label="Hipertensi"
              name="hypertensive"
              checked={form.hypertensive}
              onChange={(v) => setForm({ ...form, hypertensive: v })}
            />
            <ToggleField
              label="Riwayat Hipertensi Keluarga"
              name="familyHypertension"
              checked={form.familyHypertension}
              onChange={(v) => setForm({ ...form, familyHypertension: v })}
            />
            <ToggleField
              label="Penyakit Kardiovaskular"
              name="cardiovascular"
              checked={form.cardiovascular}
              onChange={(v) => setForm({ ...form, cardiovascular: v })}
            />
            <ToggleField
              label="Stroke"
              name="stroke"
              checked={form.stroke}
              onChange={(v) => setForm({ ...form, stroke: v })}
            />
          </div>
          <button
            type="submit"
            disabled={isLoading}
            className="rounded-lg bg-primary px-6 py-4 text-primary-foreground font-semibold disabled:opacity-50"
            aria-label="Predict My Risk"
          >
            {isLoading ? "Memprediksi..." : "Prediksi Risiko Saya"}
          </button>
        </form>

        {result != null && (
          <div className="mt-8 space-y-6">
            <div className="rounded-xl border border-border p-6 bg-card">
              <p className="text-2xl font-semibold">
                Risiko prediksi Anda adalah{" "}
                <span className={resultColor} aria-live="polite" aria-atomic="true">
                  {result.risk}%
                </span>
              </p>
              <p className="mt-3 text-muted-foreground">
                Ingat, menjaga pola makan sehat dan olahraga dapat mengurangi risiko.
              </p>
            </div>

            {shapChartData.length > 0 && (
              <div className="rounded-xl border border-border p-6 bg-card">
                <h2 className="text-xl font-semibold mb-4">Analisis Faktor Risiko</h2>
                <p className="text-sm text-muted-foreground mb-4">
                  Faktor-faktor ini paling berkontribusi pada skor risiko Anda. Bar merah meningkatkan risiko, bar hijau mengurangi risiko.
                </p>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={shapChartData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" label={{ value: 'Kontribusi', position: 'insideBottom', offset: -5 }} />
                    <YAxis type="category" dataKey="name" />
                    <Tooltip 
                      formatter={(value: number) => [`${value > 0 ? '+' : ''}${value.toFixed(3)}`, 'Dampak']}
                      contentStyle={{ backgroundColor: 'rgba(0, 0, 0, 0.8)', border: 'none', borderRadius: '8px', color: '#fff' }}
                    />
                    <Bar dataKey="contribution" radius={[0, 8, 8, 0]}>
                      {shapChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            <button
              onClick={handleAskAI}
              className="w-full rounded-lg bg-blue-600 hover:bg-blue-700 px-6 py-4 text-white font-semibold transition-colors"
              aria-label="Ask AI about my results"
            >
              💬 Tanya AI Tentang Hasil Saya
            </button>
          </div>
        )}
      </main>
      <Footer />
    </div>
  )
}
