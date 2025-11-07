class EmailRequest(PBaseModel):
    subject: Optional[str] = None
    body: str

class PredictRequest(EmailRequest):
    model: Optional[int] = None
