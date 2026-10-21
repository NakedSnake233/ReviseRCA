# Kind node image with a high inherited file-descriptor limit.  The default
# kindest/node limit (1024) is too small for four Train Ticket copies plus
# Istio/observability services.
FROM kindest/node:v1.30.0
COPY tosem_four_copy_node_entrypoint.sh /usr/local/bin/tosem-four-copy-entrypoint
RUN chmod 0755 /usr/local/bin/tosem-four-copy-entrypoint
ENTRYPOINT ["/usr/local/bin/tosem-four-copy-entrypoint", "/sbin/init"]
