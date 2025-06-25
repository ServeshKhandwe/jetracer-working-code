; Auto-generated. Do not edit!


(cl:in-package custom_msgs-srv)


;//! \htmlinclude ChatPrompt-request.msg.html

(cl:defclass <ChatPrompt-request> (roslisp-msg-protocol:ros-message)
  ((prompt
    :reader prompt
    :initarg :prompt
    :type cl:string
    :initform ""))
)

(cl:defclass ChatPrompt-request (<ChatPrompt-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ChatPrompt-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ChatPrompt-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name custom_msgs-srv:<ChatPrompt-request> is deprecated: use custom_msgs-srv:ChatPrompt-request instead.")))

(cl:ensure-generic-function 'prompt-val :lambda-list '(m))
(cl:defmethod prompt-val ((m <ChatPrompt-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader custom_msgs-srv:prompt-val is deprecated.  Use custom_msgs-srv:prompt instead.")
  (prompt m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ChatPrompt-request>) ostream)
  "Serializes a message object of type '<ChatPrompt-request>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'prompt))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'prompt))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ChatPrompt-request>) istream)
  "Deserializes a message object of type '<ChatPrompt-request>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'prompt) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'prompt) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ChatPrompt-request>)))
  "Returns string type for a service object of type '<ChatPrompt-request>"
  "custom_msgs/ChatPromptRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ChatPrompt-request)))
  "Returns string type for a service object of type 'ChatPrompt-request"
  "custom_msgs/ChatPromptRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ChatPrompt-request>)))
  "Returns md5sum for a message object of type '<ChatPrompt-request>"
  "e0e4e53d8b56a3ee57408d4081b2c754")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ChatPrompt-request)))
  "Returns md5sum for a message object of type 'ChatPrompt-request"
  "e0e4e53d8b56a3ee57408d4081b2c754")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ChatPrompt-request>)))
  "Returns full string definition for message of type '<ChatPrompt-request>"
  (cl:format cl:nil "string prompt~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ChatPrompt-request)))
  "Returns full string definition for message of type 'ChatPrompt-request"
  (cl:format cl:nil "string prompt~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ChatPrompt-request>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'prompt))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ChatPrompt-request>))
  "Converts a ROS message object to a list"
  (cl:list 'ChatPrompt-request
    (cl:cons ':prompt (prompt msg))
))
;//! \htmlinclude ChatPrompt-response.msg.html

(cl:defclass <ChatPrompt-response> (roslisp-msg-protocol:ros-message)
  ((response
    :reader response
    :initarg :response
    :type cl:string
    :initform ""))
)

(cl:defclass ChatPrompt-response (<ChatPrompt-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ChatPrompt-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ChatPrompt-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name custom_msgs-srv:<ChatPrompt-response> is deprecated: use custom_msgs-srv:ChatPrompt-response instead.")))

(cl:ensure-generic-function 'response-val :lambda-list '(m))
(cl:defmethod response-val ((m <ChatPrompt-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader custom_msgs-srv:response-val is deprecated.  Use custom_msgs-srv:response instead.")
  (response m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ChatPrompt-response>) ostream)
  "Serializes a message object of type '<ChatPrompt-response>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'response))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'response))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ChatPrompt-response>) istream)
  "Deserializes a message object of type '<ChatPrompt-response>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'response) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'response) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ChatPrompt-response>)))
  "Returns string type for a service object of type '<ChatPrompt-response>"
  "custom_msgs/ChatPromptResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ChatPrompt-response)))
  "Returns string type for a service object of type 'ChatPrompt-response"
  "custom_msgs/ChatPromptResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ChatPrompt-response>)))
  "Returns md5sum for a message object of type '<ChatPrompt-response>"
  "e0e4e53d8b56a3ee57408d4081b2c754")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ChatPrompt-response)))
  "Returns md5sum for a message object of type 'ChatPrompt-response"
  "e0e4e53d8b56a3ee57408d4081b2c754")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ChatPrompt-response>)))
  "Returns full string definition for message of type '<ChatPrompt-response>"
  (cl:format cl:nil "string response~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ChatPrompt-response)))
  "Returns full string definition for message of type 'ChatPrompt-response"
  (cl:format cl:nil "string response~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ChatPrompt-response>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'response))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ChatPrompt-response>))
  "Converts a ROS message object to a list"
  (cl:list 'ChatPrompt-response
    (cl:cons ':response (response msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'ChatPrompt)))
  'ChatPrompt-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'ChatPrompt)))
  'ChatPrompt-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ChatPrompt)))
  "Returns string type for a service object of type '<ChatPrompt>"
  "custom_msgs/ChatPrompt")