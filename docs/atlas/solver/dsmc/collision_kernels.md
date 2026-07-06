# DSMC Collision Models: Hard Sphere, VHS, and VSS

This document summarizes the theoretical differences between the **Hard Sphere (HS)**, **Variable Hard Sphere (VHS)**, and **Variable Soft Sphere (VSS)** collision models used in DSMC simulations.

The explanation is written with the provided C++ implementation in mind, but the main focus is on the underlying collision theory rather than implementation details.

---

## 1. Role of a Collision Model in DSMC

In a DSMC simulation, a collision model mainly determines two things:

1. **Collision probability or collision frequency**
2. **Post-collision scattering direction**

The collision probability is usually controlled by the total collision cross section:

$$
\sigma_T
$$

or by the product:

$$
\sigma_T g
$$

where the relative speed is:

$$
g = |\mathbf{v}_i - \mathbf{v}_j|
$$

The scattering model determines how the post-collision relative velocity direction is sampled. For elastic collisions, the magnitude of the relative velocity is preserved, while its direction is randomized according to the scattering law.

A concise summary is:

```text
HS  : constant collision cross section + isotropic scattering
VHS : speed-dependent collision cross section + isotropic scattering
VSS : speed-dependent collision cross section + alpha-controlled scattering
```

---

## 2. What Is Added from HS to VHS to VSS?

The three models can be understood as a hierarchy. Each model adds one additional physical feature compared with the previous one.

```text
HS
 |
 | adds relative-speed-dependent collision cross section
 v
VHS
 |
 | adds alpha-controlled angular scattering
 v
VSS
```

### 2.1 HS: Basic Rigid-Sphere Collision

The Hard Sphere model assumes molecules behave like rigid spheres with a fixed collision diameter.

It includes:

```text
- constant collision cross section
- isotropic post-collision scattering
- elastic conservation of momentum and kinetic energy
```

It does **not** include:

```text
- relative-speed-dependent collision cross section
- temperature-dependent viscosity correction
- adjustable angular scattering
- improved diffusion-property matching
```

Therefore, HS is the simplest collision model.

---

### 2.2 VHS: Adds Relative-Speed-Dependent Collision Cross Section

The Variable Hard Sphere model adds a velocity-dependent collision cross section to the HS model.

Compared with HS, VHS adds:

```text
- dependence of collision cross section on relative speed
- viscosity-index parameter omega
- ability to reproduce viscosity-temperature behavior
```

However, VHS still uses:

```text
- isotropic scattering
```

Therefore, VHS can be interpreted as:

```text
VHS = HS scattering + speed-dependent collision cross section
```

The main physical improvement is that VHS can model the temperature dependence of gas viscosity:

$$
\mu(T) \propto T^\omega
$$

where $\omega$ is the viscosity index.

---

### 2.3 VSS: Adds Alpha-Controlled Angular Scattering

The Variable Soft Sphere model extends VHS by adding an adjustable scattering-angle distribution.

Compared with VHS, VSS adds:

```text
- scattering parameter alpha
- non-isotropic angular scattering when alpha is not 1
- improved control of transport properties
- better modeling of diffusion and Schmidt-number behavior
```

VSS keeps the VHS-style total collision cross section but changes the angular scattering law.

Therefore, VSS can be interpreted as:

```text
VSS = VHS collision cross section + alpha-controlled angular scattering
```

When:

$$
\alpha = 1
$$

VSS reduces to VHS-style isotropic scattering.

---

## 3. Common Elastic-Collision Kinematics

All three models share the same basic elastic-collision velocity reconstruction.

Let the pre-collision particle velocities be:

$$
\mathbf{v}_i, \quad \mathbf{v}_j
$$

and the molecular masses be:

$$
m_i, \quad m_j
$$

The relative velocity is:

$$
\mathbf{g} = \mathbf{v}_i - \mathbf{v}_j
$$

with magnitude:

$$
g = |\mathbf{g}|
$$

The center-of-mass velocity is:

$$
\mathbf{V}_{cm} = \frac{m_i \mathbf{v}_i + m_j \mathbf{v}_j}{m_i + m_j}
$$

For an elastic collision:

$$
|\mathbf{g}'| = |\mathbf{g}|
$$

Only the direction of the relative velocity changes. The post-collision velocities are reconstructed as:

$$
\begin{aligned}
\mathbf{v}_i' &= \mathbf{V}_{cm} + \frac{m_j}{m_i + m_j}\mathbf{g}', \\
\mathbf{v}_j' &= \mathbf{V}_{cm} - \frac{m_i}{m_i + m_j}\mathbf{g}'.
\end{aligned}
$$

This corresponds to the following code pattern:

```cpp
const Vector3<T> center =
    (lhs_velocity * lhs_mass + rhs_velocity * rhs_mass) / mass_sum;

lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
```

Therefore, HS, VHS, and VSS do **not** mainly differ in the final velocity reconstruction. They differ in:

```text
1. how the collision cross section is computed
2. how the scattering angle is sampled
```

---

## 4. Hard Sphere Model, HS

### 4.1 Basic Idea

The Hard Sphere model treats molecules as rigid spheres with a fixed collision diameter. The total collision cross section is constant:

$$
\sigma_{HS} = \pi d^2
$$

For a mixed-species collision, an effective pair diameter is often computed as the arithmetic mean:

$$
d_{ij} = \frac{d_i + d_j}{2}
$$

Therefore:

$$
\sigma_{HS,ij} = \pi d_{ij}^2
$$

In the provided implementation:

```cpp
const T diameter =
    (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * T(0.5);

return static_cast<T>(std::numbers::pi_v<double>) * diameter * diameter;
```

---

### 4.2 Scattering Law

The HS model uses isotropic scattering. This means that the post-collision relative velocity direction is sampled uniformly on the unit sphere.

The polar scattering angle is sampled using:

$$
\cos \chi = 2R_1 - 1
$$

The azimuthal angle is sampled using:

$$
\phi = 2\pi R_2
$$

where:

$$
R_1, R_2 \in [0,1]
$$

are uniform random numbers.

In code, this is represented as:

```cpp
const T cos_chi = T(2) * u1 - T(1);
const T phi     = T(2) * pi * u2;
```

---

### 4.3 Characteristics Added by HS

HS provides the minimum physical structure needed for DSMC elastic collisions:

```text
- finite collision diameter
- constant collision cross section
- isotropic scattering
- conservation of momentum
- conservation of kinetic energy
```

This makes HS useful as a baseline model.

---

### 4.4 Limitations

HS does not model relative-speed-dependent molecular interactions.

Limitations:

```text
- collision cross section is independent of relative speed
- viscosity-temperature dependence is not accurately represented
- diffusion behavior is not specifically controlled
- real-gas transport properties may be poorly reproduced
```

Typical use cases:

```text
- baseline collision-model testing
- conservation-law verification
- simple DSMC benchmarks
- debugging of the collision pipeline
```

---

## 5. Variable Hard Sphere Model, VHS

### 5.1 Basic Idea

The Variable Hard Sphere model extends HS by allowing the collision cross section to depend on the relative speed.

For HS:

$$
\sigma = \text{constant}
$$

For VHS:

$$
\sigma = \sigma(g)
$$

The main purpose of VHS is to reproduce the temperature dependence of gas viscosity more accurately.

For many gases, the viscosity approximately follows:

$$
\mu(T) \propto T^\omega
$$

where $\omega$ is the viscosity index.

---

### 5.2 What VHS Adds Compared with HS

VHS adds the following physical features:

```text
- speed-dependent total collision cross section
- viscosity-index parameter omega
- reference temperature T_ref
- reference diameter d_ref
- improved viscosity-temperature behavior
```

The key added parameter is:

$$
\omega
$$

This parameter controls how strongly the collision cross section changes with relative speed.

However, VHS does **not** add anisotropic scattering. The scattering remains isotropic, just like HS.

---

### 5.3 VHS Collision Cross Section

The provided code implements the VHS total collision cross section as:

$$
\sigma_{VHS} = \pi d_{ref}^{2} \frac{\left(\frac{2 k_B T_{ref}}{m_r g^2}\right)^{\omega - 1/2}}{\Gamma(2.5 - \omega)}
$$

where the reduced mass is:

$$
m_r = \frac{m_i m_j}{m_i + m_j}
$$

and the relative speed is:

$$
g = |\mathbf{v}_i - \mathbf{v}_j|
$$

The corresponding code is:

```cpp
const T reduced_mass = lhs_mass * rhs_mass / mass_sum;

const T thermal_ratio =
    (T(2) * static_cast<T>(atlas::boltzmann_constant) * reference_temperature)
    / (reduced_mass * relative_speed * relative_speed);

const T reference_area =
    static_cast<T>(std::numbers::pi_v<double>)
    * reference_diameter
    * reference_diameter;

return reference_area
    * std::pow(thermal_ratio, viscosity_index - T(0.5))
    / gamma_value;
```

Because:

$$
\frac{2 k_B T_{ref}}{m_r g^2} \propto g^{-2}
$$

we can write the speed dependence as:

$$
\sigma_{VHS} \propto g^{-2(\omega - 1/2)}
$$

or equivalently:

$$
\sigma_{VHS} \propto g^{1 - 2\omega}
$$

This is the key difference between HS and VHS.

---

### 5.4 Relation Between VHS and HS

If:

$$
\omega = \frac{1}{2}
$$

then:

$$
\omega - \frac{1}{2} = 0
$$

Therefore:

$$
\left(\frac{2 k_B T_{ref}}{m_r g^2}\right)^{\omega - 1/2} = 1
$$

In this case, the VHS cross section becomes approximately constant:

$$
\sigma_{VHS} \approx \pi d_{ref}^2
$$

Thus, when:

$$
\omega = 0.5
$$

VHS behaves like a hard-sphere model.

---

### 5.5 VHS Scattering Law

Although VHS changes the collision cross section, it still uses hard-sphere-style isotropic scattering.

In the provided implementation:

```cpp
const T scattering_parameter = T(1);

const T cos_chi =
    T(2) * std::pow(u1, T(1) / scattering_parameter) - T(1);
```

Since:

$$
\alpha = 1
$$

this becomes:

$$
\cos \chi = 2u_1 - 1
$$

Therefore, VHS has the same angular scattering distribution as HS.

A stable Markdown summary is:

> **VHS = speed-dependent collision cross section + isotropic scattering**

---

### 5.6 Characteristics

Advantages:

```text
- more realistic than HS for transport-property modeling
- can reproduce viscosity-temperature behavior
- still relatively simple to implement
- commonly used as a practical DSMC collision model
```

Limitations:

```text
- angular scattering remains isotropic
- mainly corrects viscosity behavior
- may not accurately reproduce diffusion coefficients
- may not accurately reproduce Schmidt numbers
- may be insufficient for gas mixtures where species diffusion is important
```

---

## 6. Variable Soft Sphere Model, VSS

### 6.1 Basic Idea

The Variable Soft Sphere model extends VHS.

VHS modifies the collision cross section but keeps isotropic scattering. VSS keeps the VHS-style speed-dependent collision cross section and additionally modifies the angular scattering law using a scattering parameter:

$$
\alpha
$$

A stable Markdown summary is:

> **VSS = VHS collision cross section + alpha-controlled angular scattering**

In the current implementation, VSS reuses the VHS cross section:

```cpp
return VariableHardSphereKernel<T>::cross_section(lhs, rhs, relative_speed);
```

This is theoretically consistent because VSS and VHS share the same total collision cross-section structure. The difference is the scattering-angle distribution.

---

### 6.2 What VSS Adds Compared with VHS

VSS adds the following physical features:

```text
- scattering parameter alpha
- adjustable polar scattering-angle distribution
- non-isotropic scattering when alpha is not 1
- improved diffusion-property control
- improved mixture-transport modeling
- better control of Schmidt-number behavior
```

The key added parameter is:

$$
\alpha
$$

This parameter controls how the post-collision relative velocity direction is distributed.

---

### 6.3 VSS Scattering Law

The VSS polar scattering angle is sampled as:

$$
\cos \chi = 2R^{1/\alpha} - 1
$$

where:

- $\chi$ is the polar scattering angle.
- $R$ is a uniform random number in $[0,1]$.
- $\alpha$ is the VSS scattering parameter.

The code implements this as:

```cpp
const T cos_chi =
    T(2) * std::pow(u1, T(1) / scattering_parameter) - T(1);
```

The current implementation computes the pair scattering parameter using the arithmetic mean:

$$
\alpha_{ij} = \frac{\alpha_i + \alpha_j}{2}
$$

In code:

```cpp
const T scattering_parameter =
    (lhs.scattering_parameter.value_or(T(1))
     + rhs.scattering_parameter.value_or(T(1))) * T(0.5);
```

If:

$$
\alpha = 1
$$

then:

$$
\cos \chi = 2R - 1
$$

Therefore:

> **When alpha is 1, VSS reduces to VHS-style isotropic scattering.**

When $\alpha \ne 1$, the scattering-angle distribution is no longer the same as isotropic hard-sphere scattering.

---

### 6.4 Physical Meaning of VSS

VSS is designed to improve transport-property matching beyond VHS.

VHS mainly controls the viscosity-temperature relation:

$$
\mu(T) \propto T^\omega
$$

VSS introduces the additional parameter $\alpha$ to better control angular scattering. This affects transport quantities such as:

```text
- viscosity
- diffusion coefficient
- Schmidt number
- mixture transport behavior
```

This is especially important for gas mixtures, where matching viscosity alone may not be sufficient. Species diffusion can be sensitive to the angular scattering law, and VSS provides an additional degree of freedom to tune this behavior.

---

## 7. Model Comparison

| Model | Added Physical Feature | Collision Cross Section | Scattering Law | Main Parameters | Main Purpose |
|---|---|---|---|---|---|
| HS | Rigid-sphere elastic collision | Constant | Isotropic | $d$ | Simple baseline collision model |
| VHS | Relative-speed-dependent cross section | Depends on $g$ | Isotropic | $d_{ref}$, $T_{ref}$, $\omega$ | Match viscosity-temperature behavior |
| VSS | Adjustable angular scattering | Same as VHS | Controlled by $\alpha$ | $d_{ref}$, $T_{ref}$, $\omega$, $\alpha$ | Improve diffusion and mixture transport |

---

## 8. Formula Summary

### 8.1 Hard Sphere, HS

$$
\sigma_{HS} = \pi d_{ij}^{2}
$$

$$
d_{ij} = \frac{d_i + d_j}{2}
$$

$$
\cos \chi = 2R_1 - 1
$$

$$
\phi = 2\pi R_2
$$

---

### 8.2 Variable Hard Sphere, VHS

$$
\sigma_{VHS} = \pi d_{ref}^{2} \frac{\left(\frac{2k_B T_{ref}}{m_r g^2}\right)^{\omega - 1/2}}{\Gamma(2.5 - \omega)}
$$

$$
m_r = \frac{m_i m_j}{m_i + m_j}
$$

$$
\cos \chi = 2R_1 - 1
$$

$$
\phi = 2\pi R_2
$$

---

### 8.3 Variable Soft Sphere, VSS

$$
\sigma_{VSS} = \sigma_{VHS}
$$

$$
\cos \chi = 2R_1^{1/\alpha} - 1
$$

$$
\phi = 2\pi R_2
$$

$$
\alpha = 1 \Rightarrow \text{VSS reduces to VHS}
$$

---

## 9. Interpretation of the Current Code

The implementation follows this structure:

```text
HardSphereKernel
 ├─ cross_section:
 │      sigma = pi * d^2
 └─ scattering:
        isotropic

VariableHardSphereKernel
 ├─ cross_section:
 │      Bird VHS velocity-dependent cross section
 └─ scattering:
        isotropic, alpha = 1

VariableSoftSphereKernel
 ├─ cross_section:
 │      same as VHS
 └─ scattering:
        VSS angular scattering using alpha
```

This design is physically consistent.

In particular, the following VSS implementation is appropriate:

```cpp
VariableSoftSphereKernel<T>::cross_section(...)
{
    return VariableHardSphereKernel<T>::cross_section(...);
}
```

because VSS uses the VHS-style total collision cross section and modifies only the angular scattering law.

---

## 10. Model Selection Guidelines

### 10.1 When to Use HS

HS is suitable for:

```text
- testing the DSMC collision pipeline
- validating momentum and energy conservation
- building a simple baseline model
- running simple benchmark problems
```

HS is not ideal when accurate real-gas transport properties are required.

---

### 10.2 When to Use VHS

VHS is suitable for:

```text
- simulations where viscosity-temperature dependence matters
- standard practical DSMC simulations
- single-species gases or simple gas mixtures
- cases where implementation simplicity is still important
```

VHS is usually more realistic than HS and is commonly used in DSMC.

---

### 10.3 When to Use VSS

VSS is suitable for:

```text
- gas-mixture diffusion problems
- simulations where diffusion coefficients matter
- cases where the Schmidt number is important
- high-fidelity rarefied-gas transport modeling
- cases where angular scattering affects macroscopic transport behavior
```

For research simulations, a useful hierarchy is:

```text
HS  = baseline collision model
VHS = standard DSMC model for viscosity correction
VSS = improved transport model with angular scattering correction
```

---

## 11. Important Notes on the Implementation

### 11.1 HS Does Not Require Relative Speed in `cross_section`

The HS cross section is:

$$
\sigma_{HS} = \pi d^2
$$

It does not depend on relative speed. Therefore, the HS function naturally has the form:

```cpp
cross_section(lhs, rhs)
```

whereas VHS and VSS require:

```cpp
cross_section(lhs, rhs, relative_speed)
```

This distinction is physically correct.

---

### 11.2 VSS Correctly Reuses the VHS Cross Section

The VSS implementation uses:

```cpp
return VariableHardSphereKernel<T>::cross_section(lhs, rhs, relative_speed);
```

This is consistent with:

$$
\sigma_{VSS} = \sigma_{VHS}
$$

The difference between VHS and VSS is not the total collision cross section, but the angular scattering law.

---

### 11.3 VHS Uses `scattering_parameter = 1`

In the VHS operator:

```cpp
const T scattering_parameter = T(1);
```

and:

```cpp
const T cos_chi =
    T(2) * std::pow(u1, T(1) / scattering_parameter) - T(1);
```

Since:

$$
\alpha = 1
$$

this gives:

$$
\cos \chi = 2u_1 - 1
$$

Therefore, VHS uses isotropic scattering, the same angular distribution as HS.

---

### 11.4 The Key Difference in VSS Is `scattering_parameter`

The essential VSS-specific part is:

```cpp
const T scattering_parameter =
    (lhs.scattering_parameter.value_or(T(1))
     + rhs.scattering_parameter.value_or(T(1))) * T(0.5);
```

This means:

$$
\alpha_{ij} = \frac{\alpha_i + \alpha_j}{2}
$$

If no scattering parameter is provided, the code falls back to:

$$
\alpha = 1
$$

which means isotropic scattering. Therefore, in the absence of species-specific VSS parameters, the VSS kernel naturally reduces to VHS-style scattering.

---

## 12. Final Summary

The three models differ as follows:

```text
HS:
    constant cross section
    isotropic scattering
    simplest collision model

VHS:
    relative-speed-dependent cross section
    isotropic scattering
    improves viscosity-temperature behavior

VSS:
    same total cross section as VHS
    alpha-controlled angular scattering
    improves transport-property matching, especially diffusion behavior
```

The core formulas are:

$$
\sigma_{HS} = \pi d^2
$$

$$
\sigma_{VHS} = \pi d_{ref}^{2} \frac{\left(\frac{2k_B T_{ref}}{m_r g^2}\right)^{\omega - 1/2}}{\Gamma(2.5 - \omega)}
$$

$$
\sigma_{VSS} = \sigma_{VHS}, \qquad \cos \chi = 2R^{1/\alpha} - 1
$$

For the current code, the most concise explanation is:

> `HardSphereKernel` implements a constant hard-sphere collision cross section with isotropic scattering.  
> `VariableHardSphereKernel` implements a Bird-type VHS collision cross section that depends on relative speed, but still uses isotropic scattering.  
> `VariableSoftSphereKernel` reuses the VHS total collision cross section and introduces the VSS scattering parameter $\alpha$ to modify the angular scattering distribution.