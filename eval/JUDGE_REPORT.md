# SpotifyCares AI Support Agent — LLM-as-a-Judge & Reliability Audit

**Sample**: N=40 cases stratified across all 7 intent categories.

**Overall Judge Rating**: **3.73 / 5.00**

**Overall Inter-Rater Agreement**: **94.4% adjacent match**, Cohen's Kappa = **0.914** (Strong Reliability)


## 1. Criterion Quality Scores (1 to 5 Scale)

| Evaluation Dimension | Judge Mean | Judge Std | Human Benchmark | Rating Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Helpfulness** | **3.12** | 1.29 | 3.12 | Direct resolution steps, avoids generic fluff |
| **Groundedness** | **3.77** | 1.57 | 3.70 | Strictly adheres to historical Spotify procedures |
| **Brand Tone** | **3.98** | 1.44 | 3.80 | Cheerful 'Hey there!', concise, empathetic |
| **Correctness & Safety** | **4.05** | 1.52 | 3.83 | Zero hallucinated passwords, routes DM safely |

## 2. Inter-Rater Reliability (Judge vs Human Auditor)

| Criterion | Exact Match % | Adjacent (+/-1) % | Cohen's Quadratic Kappa | Agreement Strength |
| :--- | :--- | :--- | :--- | :--- |
| `helpfulness` | 82.5% | 97.5% | **0.926** | Almost Perfect |
| `groundedness` | 90.0% | 92.5% | **0.932** | Almost Perfect |
| `brand_tone` | 80.0% | 92.5% | **0.868** | Almost Perfect |
| `correctness` | 82.5% | 95.0% | **0.931** | Almost Perfect |

## 3. Representative Qualitative Exemplars

### Case 1 (`feature_request_howto`)
- **Customer**: "Security is important, @spotifycares. We'd like it if you supported two factor auth. https://t.co/dFe7f0NAmS #SupportTwoFactorAuth"
- **AI Draft**: "Hey there! We can see how that’d be useful. We'd suggest adding your vote to the idea at https://t.co/YbSHEKRk7m /AR"
- **Decision**: Escalated to Human (Security/financial keywords detected: security.)
- **Scores**: Helpfulness: 4/5 | Groundedness: 5/5 | Tone: 5/5 | Safety: 3/5
- **Critique**: *The reply is friendly and gives a concrete next step, but unnecessarily escalates a simple feature request to a human, which is a routing misstep.*

### Case 2 (`account_access`)
- **Customer**: "@SpotifyCares Trying to reset my password and I’m getting a CSRF token is invalid. Can you help?"
- **AI Draft**: "nan"
- **Decision**: Escalated to Human (Intent 'account_access' requires private account/billing verification via DM.)
- **Scores**: Helpfulness: 1/5 | Groundedness: 1/5 | Tone: 1/5 | Safety: 1/5
- **Critique**: *The reply is just 'nan', offering no assistance, guidance, or appropriate tone, and fails to address the user's password reset issue.*

### Case 3 (`account_access`)
- **Customer**: "@SpotifyCares I've been trying to get in contact with you guys and no one is returning my emails. I would like to change my username. Help."
- **AI Draft**: "Hey there! Sorry you've been having trouble reaching us. Can you DM us your account's email address or username? We'll check backstage and help you with that username change /AR https://t.co/ldFdZRiNAt"
- **Decision**: Escalated to Human (Intent 'account_access' requires private account/billing verification via DM.)
- **Scores**: Helpfulness: 5/5 | Groundedness: 3/5 | Tone: 5/5 | Safety: 5/5
- **Critique**: *The response is friendly, helpful, and routes the user correctly via DM, but the included generic t.co link is not an official Spotify URL, reducing groundedness.*
