"""Identity resolution: joins transactions/locations/sms/mails to citizens.

users.json carries no ID field at all. Empirically (verified against the
actual data, not assumed):

  * `iban` is the reliable join key between users.json and
    transactions.csv's sender_iban / recipient_iban.
  * The citizen-style sender_id/recipient_id values used in
    transactions.csv (e.g. "GRSC-KRLH-807-DIE-1") are textually identical
    to the `biotag` field in locations.json for the same person -- so once
    we know a citizen's IBAN, we also know their transaction-graph node ID
    and their GPS biotag, because they're the same string.
  * sms.json/mails.json have no ID field either. SMS are matched by phone
    number (stable per person across their thread); mails are matched by
    the "To:" display name, which matches users.json first/last name
    exactly in every sample checked.

Non-citizen counterparties (employers "EMP12345", landlords "ABIT12345",
merchants, etc.) never resolve to a profile -- that's expected, not a bug;
those IDs simply have no behavioral/geo/comm signal available and downstream
agents treat them as neutral/unknown rather than erroring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .data_loading import normalize_text
from .types import DatasetBundle

_SMS_TO_RE = re.compile(r"To:\s*(\+?\d+)")
_SMS_DATE_RE = re.compile(r"Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9:]+)")
_SMS_BODY_RE = re.compile(r"Message:\s*(.+)", re.DOTALL)
_MAIL_TO_RE = re.compile(r"To:\s*\"?([^\"<\n]+?)\"?\s*<([^>]+)>")
_MAIL_DATE_RE = re.compile(r"^Date:\s*(.+)$", re.MULTILINE)


def _full_name(user: dict[str, Any]) -> str:
    return normalize_text(f"{user.get('first_name', '')} {user.get('last_name', '')}")


@dataclass
class IdentityGraph:
    bundle: DatasetBundle
    iban_to_user: dict[str, dict[str, Any]] = field(default_factory=dict)
    citizen_to_user: dict[str, dict[str, Any]] = field(default_factory=dict)
    name_to_user: dict[str, dict[str, Any]] = field(default_factory=dict)
    phone_to_user: dict[str, dict[str, Any]] = field(default_factory=dict)
    locations_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    sms_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    mails_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def get_user(self, citizen_id: str) -> dict[str, Any] | None:
        return self.citizen_to_user.get(str(citizen_id))

    def is_profiled(self, citizen_id: str) -> bool:
        return str(citizen_id) in self.citizen_to_user

    def get_locations(self, citizen_id: str) -> pd.DataFrame:
        if self.locations_df.empty:
            return self.locations_df
        return self.locations_df[self.locations_df["biotag"] == str(citizen_id)]

    def get_sms(self, citizen_id: str) -> pd.DataFrame:
        if self.sms_df.empty:
            return self.sms_df
        return self.sms_df[self.sms_df["citizen_id"] == str(citizen_id)]

    def get_mails(self, citizen_id: str) -> pd.DataFrame:
        if self.mails_df.empty:
            return self.mails_df
        return self.mails_df[self.mails_df["citizen_id"] == str(citizen_id)]


def build_identity_graph(bundle: DatasetBundle) -> IdentityGraph:
    graph = IdentityGraph(bundle=bundle)

    users = bundle.users
    iban_to_user = {u["iban"]: u for u in users if u.get("iban")}
    name_to_user = {_full_name(u): u for u in users if u.get("first_name")}
    graph.iban_to_user = iban_to_user
    graph.name_to_user = name_to_user

    tx = bundle.transactions
    citizen_to_user: dict[str, dict[str, Any]] = {}
    for _, row in tx.iterrows():
        s_iban, r_iban = row["sender_iban"], row["recipient_iban"]
        if s_iban in iban_to_user and row["sender_id"]:
            citizen_to_user[str(row["sender_id"])] = iban_to_user[s_iban]
        if r_iban in iban_to_user and row["recipient_id"]:
            citizen_to_user[str(row["recipient_id"])] = iban_to_user[r_iban]
    graph.citizen_to_user = citizen_to_user

    # locations.json: biotag IS the citizen_id (verified empirically).
    if bundle.locations:
        loc_df = pd.DataFrame(bundle.locations)
        loc_df["biotag"] = loc_df["biotag"].astype(str)
        loc_df["timestamp"] = pd.to_datetime(loc_df["timestamp"], errors="coerce")
        loc_df["lat"] = pd.to_numeric(loc_df["lat"], errors="coerce")
        loc_df["lng"] = pd.to_numeric(loc_df["lng"], errors="coerce")
        graph.locations_df = loc_df.sort_values("timestamp").reset_index(drop=True)

    # sms.json: bucket by phone number, resolve each bucket to a user by
    # the first name mentioned inside "Message: <Name>: ...".
    if bundle.sms:
        known_first_names = {
            normalize_text(u["first_name"]).split(" ")[0]
            for u in users if u.get("first_name")
        }
        rows = []
        phone_name_votes: dict[str, dict[str, int]] = {}
        for item in bundle.sms:
            text = item.get("sms", "")
            phone_m = _SMS_TO_RE.search(text)
            date_m = _SMS_DATE_RE.search(text)
            phone = phone_m.group(1) if phone_m else ""

            # Message body phrasing varies ("Message: Alain: ...", "Message:
            # Hi Tania, ...", alert-style with no name at all) so instead of
            # anchoring to one shape, scan the whole body for any known
            # first name as a whole word. Only vote when exactly one known
            # name appears, since an ambiguous mention (someone else's name
            # quoted inside the text) shouldn't move the tally.
            body_m = _SMS_BODY_RE.search(text)
            body_norm = normalize_text(body_m.group(1)) if body_m else ""
            mentioned = {
                fname for fname in known_first_names
                if re.search(rf"\b{re.escape(fname)}\b", body_norm)
            }
            name = next(iter(mentioned)) if len(mentioned) == 1 else ""
            if phone and name:
                votes = phone_name_votes.setdefault(phone, {})
                votes[name] = votes.get(name, 0) + 1
            rows.append({
                "raw": text,
                "phone": phone,
                "timestamp": pd.to_datetime(date_m.group(1), errors="coerce") if date_m else pd.NaT,
            })

        phone_to_user: dict[str, dict[str, Any]] = {}
        for phone, votes in phone_name_votes.items():
            best_name = max(votes, key=votes.get)
            for full_name, user in name_to_user.items():
                if full_name.split(" ")[0] == best_name.split(" ")[0]:
                    phone_to_user[phone] = user
                    break
        graph.phone_to_user = phone_to_user

        sms_df = pd.DataFrame(rows)
        sms_df["citizen_id"] = ""
        for citizen_id, user in citizen_to_user.items():
            user_phones = {p for p, u in phone_to_user.items() if u is user}
            if user_phones:
                sms_df.loc[sms_df["phone"].isin(user_phones), "citizen_id"] = citizen_id
        graph.sms_df = sms_df.sort_values("timestamp").reset_index(drop=True)

    # mails.json: match the "To:" display name directly against users.
    if bundle.mails:
        rows = []
        for item in bundle.mails:
            text = item.get("mail", "")
            to_m = _MAIL_TO_RE.search(text)
            date_m = _MAIL_DATE_RE.search(text)
            to_name = normalize_text(to_m.group(1)) if to_m else ""
            ts = pd.NaT
            if date_m:
                ts = pd.to_datetime(date_m.group(1), errors="coerce", utc=True)
                if pd.notna(ts):
                    ts = ts.tz_localize(None)
            rows.append({"raw": text, "to_name": to_name, "timestamp": ts})

        mails_df = pd.DataFrame(rows)
        mails_df["citizen_id"] = ""
        name_to_citizen = {
            _full_name(user): citizen_id for citizen_id, user in citizen_to_user.items()
        }
        for full_name, citizen_id in name_to_citizen.items():
            mails_df.loc[mails_df["to_name"] == full_name, "citizen_id"] = citizen_id
        graph.mails_df = mails_df.sort_values("timestamp").reset_index(drop=True)

    return graph
