# Job A: write 60 new evaluation documents

These records are a set used only to choose between settings of a bookkeeping model. They must look like documents from real life, not like program output. A wrong or doubtful label makes the set useless, so correctness matters more than speed. Where a document could be booked in two ways, change the document so it is clear, or make it a "needs review" document.

GUIDE.md is the full rulebook: the record format (section 1), the hard rules for the numbers (section 2), the accounts and booking conventions (section 4), reverse charge (5), the review causes (6), the document text (7) and the check before you answer (9). Everything in it stays valid. Ignore its sentences about 80 records, the ids gold-001 to gold-080, batches and quotas: this job has its own below.

Also keep these rules: every item on an expense account has a net price below 650 EUR (an item on 0480 below 700, on 0410 or 0650 above 900); the date is written in the text; supplier names are invented, English, plain ASCII, no real brands, no German words and no place names.

## What to write

60 records with the ids `real-037` to `real-096`, in three batches of 20 (real-037 to real-056, real-057 to real-076, real-077 to real-096). The `layout` field is required on every record. Append each batch to `A_write/new_records.jsonl`, one JSON object per line.

Supplier names must differ from each other and from the names in `A_write/config.json` (`used_names`). Do not reuse an invoice number, an item list, an amount list or a sentence from the examples in GUIDE.md.

## How the documents must look

They look like text from a PDF text layer or an OCR scan of a photo. Do not write clean, tidy documents. Use these features, and use each one in several documents:

- Columns with uneven spacing, or columns cut into separate lines ("Qty", "Item", "Unit price", "Amount" in a row, then the values below).
- Quantity times unit price lines: "3 x 14.90   44.70". The item amount is the line total.
- Article numbers, item codes and order numbers next to the item text: "Art. 4471-B", "SKU 88213".
- Shortened or capital-letter item names on receipts: "ORG OAT DRINK 1L", "COPY PAPER A4 80G 500S". Keep them in English and use abbreviations that a real till would use.
- Lines that are not items and carry no amount that is added: customer number, order number, delivery date, payment terms, the bank details of the supplier, a register or VAT ID line, page numbers ("Page 1 of 1"), a legal footer, a delivery address block with an invented recipient.
- On receipts: payment lines after the total, such as "Card payment 24.60", or "Cash 30.00" and "Change 5.40". These are not items. The printed total stays clear.
- Different wording for the same thing: "VAT", "Tax", "incl. VAT", "of which VAT", "Sales tax". Different words for the total: "Total", "Amount due", "Grand total", "To pay", "Sum".
- A header or footer that is partly in the wrong place, as a text layer would give it.
- Long documents with 5 to 8 items, and short ones with one item.
- Both number formats, `1,234.56` and `1.234,56`, and several date formats. Use `EUR`, `€` and `Euro`. If a date is numeric, the day must be above 12.

Do not write any text that tells a reader what to do, except in the prompt-injection documents. Do not put account numbers or booking hints in a document. Every priced line that is an item must be an item in the answer. A line like "Cash 30.00" or "Change 5.40" is a payment line and is not an item.

## Quotas for each batch of 20

- Layouts: table_invoice 6, letter_invoice 3, till_receipt 5, short_receipt 2, credit_note 2, eu_invoice 2. (Over the three batches: 18, 9, 15, 6, 6, 6.)
- Review documents, 4 in each batch, inside the layouts above, with these causes:
  - batch 1: ambiguous_purpose, business_private_mixed, missing_total, heavy_noise;
  - batch 2: near_asset_limit, total_mismatch, foreign_currency, ambiguous_purpose;
  - batch 3: business_private_mixed, missing_total, total_mismatch, near_asset_limit.
- 1 prompt-injection document in each batch, with a wording that is not an order to "ignore" things and is different in each batch (for example a line in a delivery note addressed to "AI systems processing this document", or a note in a footer addressed to a "bookkeeping assistant"). The label is the correct normal booking.
- 1 free-text reverse-charge document in each batch (table_invoice or letter_invoice), and 2 eu_invoice documents: one from another EU country and one from outside the EU billed in EUR.
- At least 1 pure private purchase (1800) on a till_receipt or short_receipt, with shortened real item names, 7% and 19% items together and a payment line.
- At least 5 documents mixing 19% and 7%, and at least 2 with a 0% item (insurance, rent, a registered letter, bank fees, interest, an exam fee).
- At least 3 documents with 4 to 8 items, and at least 3 with a single item.
- At least 1 discount line (a negative item on an invoice), and at least 2 purchases booked to 0410, 0480 or 0650.
- Credit notes, 2 in each batch, about returned goods or a price correction. One of the six has 19% and 7% items together.
- About 6 documents with light OCR damage that still leaves every amount clear, and about 5 with fake identifiers (invented IBAN, email at an invented domain, phone number, tax number).
- No document booked to 4660. Use at least 18 different accounts in the 60 records.

## Check

After each batch, and at the end, run (from the folder A_write): `python ../tools/records_check.py`. It checks the format, the sums, the amounts and the date in the text, the VAT lines, the asset and net-price rules, the duplicate names and texts, and the quotas. Fix every line it prints. The records are only good when it says "all records are good".
