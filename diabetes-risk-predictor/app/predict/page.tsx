"use client"

import type React from "react"

import { useEffect, useMemo, useState } from "react"
import { Navbar } from "@/components/navbar"
import { Footer } from "@/components/footer"
import { InputField, SelectField, ToggleField } from "@/components/input-field"

type PredictForm = {
  age: number | ""
  gender: "Male" | "Female"
  pulseRate: number | ""
  sbp: number | ""
  dbp: number | ""
  glucose: number | ""
  heightCm: number | ""
  weightKg: number | ""
  bmi: number | ""
  familyDiabetes: boolean
  hypertensive: boolean
  familyHypertension: boolean
  cardiovascular: boolean
  stroke: boolean
}

export default function PredictPage() {
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
  const [result, setResult] = useState<number | null>(null)

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
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...form, bmi: Number(form.bmi || autoBmi || 0) }),
    })
    const data = await res.json()
    setResult(data.risk)
    // Save to history
    const history = JSON.parse(localStorage.getItem("predHistory") || "[]")
    const entry = { date: new Date().toISOString(), risk: data.risk, inputs: form }
    localStorage.setItem("predHistory", JSON.stringify([entry, ...history].slice(0, 50)))
    localStorage.setItem("lastRisk", String(data.risk))
  }

  const resultColor = result == null ? "" : result < 33 ? "text-success" : result < 66 ? "text-warning" : "text-danger"

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Navbar />
      <main className="mx-auto max-w-4xl px-4 py-8">
        <h1 className="text-3xl font-semibold mb-6">Check Your Diabetes Risk</h1>
        <form onSubmit={onSubmit} className="grid gap-5">
          <div className="grid gap-4 sm:grid-cols-2">
            <InputField
              label="Age"
              name="age"
              type="number"
              value={form.age}
              onChange={(v) => setForm({ ...form, age: v === "" ? "" : Number(v) })}
              required
            />
            <SelectField
              label="Gender"
              name="gender"
              value={form.gender}
              onChange={(v) => setForm({ ...form, gender: v as "Male" | "Female" })}
              options={[
                { value: "Male", label: "Male" },
                { value: "Female", label: "Female" },
              ]}
            />
            <InputField
              label="Pulse Rate"
              name="pulseRate"
              type="number"
              value={form.pulseRate}
              onChange={(v) => setForm({ ...form, pulseRate: v === "" ? "" : Number(v) })}
            />
            <InputField
              label="Systolic BP"
              name="sbp"
              type="number"
              value={form.sbp}
              onChange={(v) => setForm({ ...form, sbp: v === "" ? "" : Number(v) })}
            />
            <InputField
              label="Diastolic BP"
              name="dbp"
              type="number"
              value={form.dbp}
              onChange={(v) => setForm({ ...form, dbp: v === "" ? "" : Number(v) })}
            />
            <InputField
              label="Glucose"
              name="glucose"
              type="number"
              value={form.glucose}
              onChange={(v) => setForm({ ...form, glucose: v === "" ? "" : Number(v) })}
            />
            <InputField
              label="Height (cm)"
              name="height"
              type="number"
              value={form.heightCm}
              onChange={(v) => setForm({ ...form, heightCm: v === "" ? "" : Number(v) })}
            />
            <InputField
              label="Weight (kg)"
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
              hint="Auto-calculated from height and weight; you can adjust if needed."
            />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <ToggleField
              label="Family Diabetes"
              name="familyDiabetes"
              checked={form.familyDiabetes}
              onChange={(v) => setForm({ ...form, familyDiabetes: v })}
            />
            <ToggleField
              label="Hypertensive"
              name="hypertensive"
              checked={form.hypertensive}
              onChange={(v) => setForm({ ...form, hypertensive: v })}
            />
            <ToggleField
              label="Family Hypertension"
              name="familyHypertension"
              checked={form.familyHypertension}
              onChange={(v) => setForm({ ...form, familyHypertension: v })}
            />
            <ToggleField
              label="Cardiovascular Disease"
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
            className="rounded-lg bg-primary px-6 py-4 text-primary-foreground font-semibold"
            aria-label="Predict My Risk"
          >
            Predict My Risk
          </button>
        </form>

        {result != null && (
          <div className="mt-8 rounded-xl border border-border p-6 bg-card">
            <p className="text-2xl font-semibold">
              Your predicted risk is{" "}
              <span className={resultColor} aria-live="polite" aria-atomic="true">
                {result}%
              </span>
            </p>
            <p className="mt-3 text-muted-foreground">
              Remember, maintaining healthy diet and exercise can reduce risk.
            </p>
          </div>
        )}
      </main>
      <Footer />
    </div>
  )
}
