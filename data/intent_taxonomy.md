# SpotifyCares Empirical Intent Taxonomy

Based on an empirical analysis of customer support tweets from the SpotifyCares subsample (`twcs.csv`), customer inquiries naturally cluster into **7 distinct, mutually exclusive intents**.

---

## Intent Definitions

| Intent ID | Intent Name | Description | Default Policy |
| :--- | :--- | :--- | :--- |
| `playback_streaming` | **Playback & Audio Streaming** | Customer experiences glitches, offline playback failure, songs pausing/stopping, caching issues, or device connect errors. | Auto-handle with standard troubleshooting (device/app restart, reinstall, offline toggle) unless repeated. |
| `account_access` | **Account Access & Security** | Inability to log in, password reset failures, third-party auth issues (Facebook), hacked accounts, or changing account credentials. | **Always Escalate**: Requires secure verification via private Direct Message (DM) or account credentials. |
| `billing_subscription` | **Billing, Subscription & Refunds** | Charges after cancellation, payment failures, Premium status not activating, price inquiries, or refund demands. | **Always Escalate**: Involves financial transactions, PCI compliance, and private billing records. |
| `content_availability` | **Content Availability & Licensing** | Missing songs/albums, greyed-out tracks, inquiries about new releases (e.g. Taylor Swift 'Reputation'), or regional catalog rights. | Auto-handle: Explain catalog licensing variations and suggest artist follow / notification tools. |
| `feature_request_howto` | **Feature Requests & How-To** | Questions on how to perform an action (playlists, crossfade, local files, storage management) or suggestions for new features. | Auto-handle: Provide direct step-by-step help, knowledge base links, or acknowledge feedback to devs. |
| `venting_feedback` | **Venting & Opinion (No Direct Ask)** | Complaints about algorithmic curation (Discover Weekly, Daily Mix), disgust with UI redesigns, or emotional rants without a technical request. | Auto-handle with empathetic brand tone; escalate only if abusive/extreme sentiment requires PR moderation. |
| `other` | **Other / Ambiguous / Contextless** | Incomplete messages, mid-conversation follow-ups ("thanks", "did that work?"), non-English queries, or unintelligible tweets. | Escalate if unclear ask; auto-handle with request for more context if benign. |

---

## Detailed Intent Profiles & Real Examples

### 1. `playback_streaming`
* **Description**: Technical issues with playing audio, local files syncing, offline downloads, audio cutouts, or Bluetooth/smart speaker connect.
* **Keywords**: *can't play, offline, pause, stops, buffering, downloaded, songs won't play, shuffle, connect, chromecast, speaker, sound*
* **Real Dataset Examples**:
  - *"@SpotifyCares I can’t listen to songs on Spotify offline despite being a premium user"*
  - *"@SpotifyCares @spotify - My downloaded music is unavailable offline. What's the deal? It keeps saying I need an internet connection."*
  - *"How do I get my smart speakers (@122986 home, @115833 or @115890 #cortana) to NOT default to playing the single on @SpotifyCares but the whole album."*

### 2. `account_access`
* **Description**: Account login failures, account compromised/hacked, email verification, or password reset.
* **Keywords**: *login, password, sign in, hacked, account, email, username, facebook linked, verification, credentials*
* **Real Dataset Examples**:
  - *"@SpotifyCares Says no Facebook linked, I can't us my email and Facebook doesn't work? When I try on the app it fails once I login to Facebook"*
  - *"@115888 @SpotifyCares been hacked need some help"*
  - *"@SpotifyCares account insists I'm in the US. Tried changing to UK, won't let me. Help!"*

### 3. `billing_subscription`
* **Description**: Issues involving money, charges, payment methods, upgrade to Premium, subscription cancellation, or refunds.
* **Keywords**: *charge, charged, billing, subscription, cancel, refund, premium, payment, money, card, receipt, pay*
* **Real Dataset Examples**:
  - *"@SpotifyCares I cancelled my premium subscription but you have taken payment from my phone account regardless. I need it back please"*
  - *"@SpotifyCares No se me activa mi cuenta premium que compre en el OXXO, que puedo hacer?"*
  - *"@SpotifyCares I can't access my Facebook account as it's been deleted but I'm still being charged for premium, how do I cancel?"*

### 4. `content_availability`
* **Description**: Inquiries regarding unavailable songs, missing albums, podcast episodes, or regional licensing rights.
* **Keywords**: *missing, album, song, artist, licensing, available, release, removed, greyed out, catalogue, when will*
* **Real Dataset Examples**:
  - *"Why are Anita's albums not on @115888 ?"*
  - *"why isn't f(x)'s nu abo album on @115888?"*
  - *"@115888 dudes, fix your shit and put reputation on spotify, i've been waiting for like, a week."*

### 5. `feature_request_howto`
* **Description**: Inquiries on how to use Spotify features, configure settings, or suggestions for product improvements.
* **Keywords**: *how to, how do i, feature, suggestion, setting, option, where is, add to playlist, storage, crossfade, queue*
* **Real Dataset Examples**:
  - *"@SpotifyCares Feature suggestion: ability to stream to @44842 from the PC app."*
  - *"@115888 Am I correct to conclude your Android app gives me no way to control how much storage you use? Hope I'm missing something."*
  - *"I’d like to see my @115888 play count for both Twinkle Twinkle Little Star and the album ‘2017 White Noise for Baby Sleep’"*

### 6. `venting_feedback`
* **Description**: Negative or sarcastic commentary, dissatisfaction with recommendation algorithms, or general venting without an actionable troubleshooting ask.
* **Keywords**: *hate, worst, terrible, sucks, annoyed, discover weekly, why would you, rude, trash, dislike*
* **Real Dataset Examples**:
  - *"@115888 this is the worst song i’ve ever heard, and yet you insist on putting it in my discover weekly every other week, please explain"*
  - *"@115888 why the hell would you allow a random Jonas Brother song that I don't even have saved play after my Merry Lit'mas playlist that is fucking RUDE"*

### 7. `other`
* **Description**: Conversational pleasantries, non-English messages, or fragmentary thread continuations without sufficient standalone context.
* **Keywords**: *thanks, okay, what?, hello, https://t.co/..., yes, no, link*
* **Real Dataset Examples**:
  - *"@SpotifyCares (One previous thread here https://t.co/hoQejA4HyW)"*
  - *"@SpotifyCares Nothing will get deleted or have to change will it? (Just curious)"*
  - *"@115888 when are you going to support Slovenia ? 🇸🇮"*
