FROM grafana/grafana:10.4.2

ENV GF_USERS_ALLOW_SIGN_UP=false \
    GF_PATHS_PROVISIONING=/etc/grafana/provisioning

COPY monitoring/grafana/provisioning/datasources.yaml /etc/grafana/provisioning/datasources/datasources.yaml
COPY monitoring/grafana/provisioning/dashboards.yaml /etc/grafana/provisioning/dashboards/dashboards.yaml
COPY monitoring/grafana/dashboards/ /etc/grafana/dashboards/
COPY monitoring/grafana/provisioning/alerting/ /etc/grafana/provisioning/alerting/
