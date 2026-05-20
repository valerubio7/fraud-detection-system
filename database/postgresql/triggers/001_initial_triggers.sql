DROP TRIGGER IF EXISTS alert_on_high_fraud_rate ON public.predictions_history;

CREATE TRIGGER alert_on_high_fraud_rate
AFTER INSERT ON public.predictions_history
FOR EACH ROW
EXECUTE FUNCTION public.check_fraud_rate();
