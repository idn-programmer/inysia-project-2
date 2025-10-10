"use client"

type BaseProps = {
  label: string
  name: string
  required?: boolean
  hint?: string
}

type InputProps = BaseProps & {
  type?: "text" | "email" | "password" | "number"
  value: string | number
  onChange: (v: string) => void
  placeholder?: string
  min?: number
  max?: number
  step?: number
}

export function InputField({
  label,
  name,
  type = "text",
  value,
  onChange,
  required,
  hint,
  placeholder,
  min,
  max,
  step,
}: InputProps) {
  return (
    <div className="grid gap-2">
      <label htmlFor={name} className="font-medium">
        {label}{" "}
        {required && (
          <span aria-hidden="true" className="text-danger">
            *
          </span>
        )}
      </label>
      <input
        id={name}
        name={name}
        aria-label={label}
        className="w-full rounded-lg border border-input bg-background px-4 py-3"
        type={type}
        value={String(value ?? "")}
        onChange={(e) => onChange(e.target.value)}
        required={required}
        placeholder={placeholder}
        min={min}
        max={max}
        step={step}
      />
      {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
    </div>
  )
}

type SelectFieldProps = BaseProps & {
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string }[]
}

export function SelectField({ label, name, value, onChange, required, hint, options }: SelectFieldProps) {
  return (
    <div className="grid gap-2">
      <label htmlFor={name} className="font-medium">
        {label}{" "}
        {required && (
          <span aria-hidden="true" className="text-danger">
            *
          </span>
        )}
      </label>
      <select
        id={name}
        name={name}
        aria-label={label}
        className="w-full rounded-lg border border-input bg-background px-4 py-3"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required={required}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      {hint && <p className="text-sm text-muted-foreground">{hint}</p>}
    </div>
  )
}

type ToggleFieldProps = BaseProps & {
  checked: boolean
  onChange: (v: boolean) => void
}

export function ToggleField({ label, name, checked, onChange }: ToggleFieldProps) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-input bg-background p-3">
      <label htmlFor={name} className="font-medium">
        {label}
      </label>
      <input
        id={name}
        name={name}
        type="checkbox"
        role="switch"
        aria-checked={checked}
        className="size-6 accent-primary"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
    </div>
  )
}
